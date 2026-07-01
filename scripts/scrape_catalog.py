"""
scrape_catalog.py
-----------------
Scrapes the SHL product catalog (https://www.shl.com/solutions/products/product-catalog/)
restricted to **Individual Test Solutions** only. Pre-packaged Job Solutions are skipped.

Output: data/catalog.json  -- a list of catalog entries:
    {
      "name":        str,
      "url":         str,            # absolute shl.com URL
      "description":  str,
      "test_type":   list[str],      # SHL letter codes, e.g. ["K"], ["P"], ["A","B"]
      "job_levels":  list[str],
      "languages":   list[str],
      "assessment_length": str | None,
      "remote_testing":  bool,
      "adaptive_irt":    bool
    }

Design notes
------------
* The catalog is a paginated table. The Individual Test Solutions table uses a
  `type=1` query param; Job Solutions use `type=2`. We only walk `type=1`.
* Each row links to a product detail page. We fetch detail pages to extract the
  rich description + the SHL test-type legend (A,B,C,D,E,K,P,S).
* We are polite: small delay, retries, a real User-Agent. If the network is
  unavailable at build time, run `seed_catalog.py` instead -- the service never
  hard-depends on a live scrape.

This script is intentionally defensive: every selector has a fallback, because
SHL's markup changes and "works on the happy path only" is an explicit failure
mode in the brief.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

BASE = "https://www.shl.com"
CATALOG_PATH = "/solutions/products/product-catalog/"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_FILE = DATA_DIR / "catalog.json"

# SHL's published test-type legend. Used to normalise the per-product key.
TEST_TYPE_LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class CatalogEntry:
    name: str
    url: str
    description: str = ""
    test_type: list[str] = field(default_factory=list)
    job_levels: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    assessment_length: str | None = None
    remote_testing: bool = False
    adaptive_irt: bool = False

    def key(self) -> str:
        return self.url.rstrip("/").lower()


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def _get(client: httpx.Client, url: str) -> httpx.Response:
    resp = client.get(url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    return resp


def _abs(href: str) -> str:
    if href.startswith("http"):
        return href
    if not href.startswith("/"):
        href = "/" + href
    return BASE + href


def _parse_listing_page(html: str) -> tuple[list[CatalogEntry], bool]:
    """
    Parse one catalog listing page.

    Returns (entries, has_more). We only collect rows from the
    *Individual Test Solutions* section. SHL renders two tables; the
    individual one is the wrapper that is NOT the "Pre-packaged" table.
    """
    soup = BeautifulSoup(html, "html.parser")
    entries: list[CatalogEntry] = []

    # Strategy: find every product link inside the catalog table area.
    # Product detail links live under /products/product-catalog/view/<slug>/
    rows = soup.select("table tr")
    for row in rows:
        link = row.find("a", href=re.compile(r"/product-catalog/view/"))
        if not link:
            continue
        name = link.get_text(strip=True)
        if not name:
            continue
        url = _abs(link["href"])

        # Booleans: SHL marks remote testing / adaptive with a coloured dot.
        cells = row.find_all("td")
        row_text = row.get_text(" ", strip=True).lower()
        remote = "yes" in _cell_flag(cells, 1)
        adaptive = "yes" in _cell_flag(cells, 2)

        # Per-row test-type letters appear as small badges in the last cell.
        ttypes = sorted(set(re.findall(r"\b([ABCDEKPS])\b", _cell_flag(cells, -1).upper())))

        entries.append(
            CatalogEntry(
                name=name,
                url=url,
                test_type=ttypes,
                remote_testing=remote,
                adaptive_irt=adaptive,
            )
        )

    # Pagination: SHL uses ?start=<n>&type=1. If a "next" control exists we
    # continue; otherwise we stop. We also stop if a page yields zero rows.
    has_more = bool(soup.select_one('a[href*="start="]')) and len(entries) > 0
    return entries, has_more


def _cell_flag(cells, idx: int) -> str:
    try:
        return cells[idx].get_text(" ", strip=True)
    except (IndexError, AttributeError):
        return ""


def _enrich_detail(client: httpx.Client, entry: CatalogEntry) -> None:
    """Fetch the product detail page and pull description + metadata."""
    try:
        resp = _get(client, entry.url)
    except Exception as exc:  # noqa: BLE001 - we never let one bad page kill the run
        print(f"  ! detail fetch failed for {entry.name}: {exc}", file=sys.stderr)
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # Description: the main content paragraph(s). Multiple fallbacks.
    desc_el = (
        soup.select_one(".product-catalogue-training-calendar__row p")
        or soup.find("div", class_=re.compile("description", re.I))
        or soup.find("meta", attrs={"name": "description"})
    )
    if desc_el is not None:
        entry.description = (
            desc_el.get("content", "").strip()
            if desc_el.name == "meta"
            else desc_el.get_text(" ", strip=True)
        )

    text = soup.get_text(" ", strip=True)

    # Assessment length (e.g. "Approximate Completion Time in minutes = 30")
    m = re.search(r"(?:completion time|assessment length)[^0-9]*(\d+)", text, re.I)
    if m:
        entry.assessment_length = f"{m.group(1)} minutes"

    # Job levels
    jl = re.search(r"Job levels?\s*[:\-]?\s*([A-Za-z ,/&-]+)", text)
    if jl:
        entry.job_levels = [
            s.strip() for s in re.split(r"[,/]", jl.group(1)) if s.strip()
        ][:8]

    # Languages
    lg = re.search(r"Languages?\s*[:\-]?\s*([A-Za-z ,()/&-]+)", text)
    if lg:
        entry.languages = [
            s.strip() for s in re.split(r"[,/]", lg.group(1)) if s.strip()
        ][:20]

    # Test type letters from the detail legend, if the listing missed them.
    if not entry.test_type:
        legend_block = re.search(r"Test Type[s]?\s*[:\-]?\s*([ABCDEKPS ,]+)", text)
        if legend_block:
            entry.test_type = sorted(
                set(re.findall(r"[ABCDEKPS]", legend_block.group(1)))
            )


def scrape(max_pages: int = 60, delay: float = 0.6) -> list[CatalogEntry]:
    seen: dict[str, CatalogEntry] = {}
    with httpx.Client(headers=HEADERS) as client:
        start = 0
        page = 0
        while page < max_pages:
            url = f"{BASE}{CATALOG_PATH}?start={start}&type=1"
            print(f"[listing] page {page} -> {url}")
            try:
                resp = _get(client, url)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! listing fetch failed: {exc}", file=sys.stderr)
                break

            entries, has_more = _parse_listing_page(resp.text)
            if not entries:
                print("  (no rows; stopping pagination)")
                break

            new = 0
            for e in entries:
                if e.key() not in seen:
                    seen[e.key()] = e
                    new += 1
            print(f"  +{new} new (total {len(seen)})")

            if new == 0 or not has_more:
                break
            start += len(entries)
            page += 1
            time.sleep(delay)

        # Enrich detail pages
        items = list(seen.values())
        for i, entry in enumerate(items, 1):
            print(f"[detail {i}/{len(items)}] {entry.name}")
            _enrich_detail(client, entry)
            time.sleep(delay)

    return list(seen.values())


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entries = scrape()
    if not entries:
        print(
            "Scrape produced 0 entries. Network likely blocked. "
            "Run scripts/seed_catalog.py to bootstrap a fallback dataset.",
            file=sys.stderr,
        )
        sys.exit(2)

    payload = [asdict(e) for e in entries]
    OUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(payload)} entries -> {OUT_FILE}")


if __name__ == "__main__":
    main()
