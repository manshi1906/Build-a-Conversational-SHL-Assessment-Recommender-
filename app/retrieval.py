"""
retrieval.py
------------
Hybrid retrieval over the SHL catalog.

Why hybrid
----------
Recall@10 is a major part of the score. Pure semantic search misses exact
product names ("OPQ32r", "Java 8", "GSA"); pure keyword search misses
paraphrased intent ("someone who works well with stakeholders" -> personality).
We combine:

  1. Dense semantic similarity (sentence-transformers, cosine over a small
     in-memory matrix -- no external vector DB needed for ~hundreds of items;
     this keeps the deployment a single process and well under the 30s budget).
  2. Sparse lexical scoring (BM25) for exact terms, names and acronyms.
  3. Light structured boosts from extracted constraints (test_type, job level).

Scores are min-max normalised per query then linearly fused. The fusion
weights were tuned on the public-style traces in eval/ (see approach doc).

Falls back gracefully: if sentence-transformers can't load (offline build),
retrieval degrades to BM25-only rather than crashing. "Breaks on anything
else" is an explicit failure mode we guard against.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_FILE = DATA_DIR / "catalog.json"

_WORD = re.compile(r"[a-z0-9+#.]+")


def _tok(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Doc:
    idx: int
    name: str
    url: str
    description: str
    test_type: list[str]
    job_levels: list[str]
    blob: str  # searchable text

    @property
    def tokens(self) -> list[str]:
        return _tok(self.blob)


class BM25:
    """Compact BM25 (Okapi) implementation. No external dependency."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.N, 1)
        self.df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in self.df.items()
        }
        self.tf = [self._counts(d) for d in corpus]

    @staticmethod
    def _counts(doc: list[str]) -> dict[str, int]:
        c: dict[str, int] = {}
        for t in doc:
            c[t] = c.get(t, 0) + 1
        return c

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.N
        for i in range(self.N):
            dl = len(self.corpus[i])
            s = 0.0
            for term in query:
                if term not in self.idf:
                    continue
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf[term] * (f * (self.k1 + 1)) / denom
            out[i] = s
        return out


def _minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


class Retriever:
    """
    Hybrid retriever. Build once at startup, query per turn.

    .search(query, constraints, k) -> list[dict] catalog entries, ranked.
    """

    def __init__(self, catalog_path: Path = CATALOG_FILE):
        raw = json.loads(Path(catalog_path).read_text(encoding="utf-8", errors="replace"))   
        self.docs: list[Doc] = []
        for i, r in enumerate(raw):
            blob = " ".join(
                [
                    r.get("name", ""),
                    r.get("name", ""),  # name weighted x2 for lexical match
                    r.get("description", ""),
                    " ".join(r.get("test_type", [])),
                    " ".join(r.get("job_levels", [])),
                ]
            )
            self.docs.append(
                Doc(
                    idx=i,
                    name=r.get("name", ""),
                    url=r.get("url", ""),
                    description=r.get("description", ""),
                    test_type=r.get("test_type", []),
                    job_levels=r.get("job_levels", []),
                    blob=blob,
                )
            )
        self._raw = raw
        self.bm25 = BM25([d.tokens for d in self.docs])

        # Dense embeddings -- optional. Lazy import so the service still boots
        # if the model can't be fetched in a restricted build environment.
        self._embed = None
        self._mat = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as np  # noqa: F401

            self._embed = SentenceTransformer("all-MiniLM-L6-v2")
            import numpy as np

            vecs = self._embed.encode(
                [d.blob for d in self.docs],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._mat = np.asarray(vecs, dtype="float32")
            self._np = np
        except Exception as exc:  # noqa: BLE001
            print(f"[retrieval] dense embeddings disabled ({exc}); BM25-only mode")

    # ------------------------------------------------------------------ #

    def _dense_scores(self, query: str) -> list[float] | None:
        if self._embed is None or self._mat is None:
            return None
        q = self._embed.encode([query], normalize_embeddings=True)[0]
        sims = self._mat @ q  # cosine (vectors are normalised)
        return sims.tolist()

    def search(
        self,
        query: str,
        constraints: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[dict]:
        constraints = constraints or {}
        q_tokens = _tok(query)

        lex = _minmax(self.bm25.scores(q_tokens)) if q_tokens else [0.0] * len(self.docs)
        dense_raw = self._dense_scores(query) if query.strip() else None
        sem = _minmax(dense_raw) if dense_raw is not None else [0.0] * len(self.docs)

        # Fusion weights (tuned on dev traces). If dense is unavailable we lean
        # fully on lexical so behaviour stays sane.
        w_sem, w_lex = (0.6, 0.4) if dense_raw is not None else (0.0, 1.0)

        wanted_types = {t.upper() for t in constraints.get("test_type", [])}
        wanted_level = (constraints.get("job_level") or "").lower()

        scored: list[tuple[float, Doc]] = []
        for d in self.docs:
            score = w_sem * sem[d.idx] + w_lex * lex[d.idx]

            # Structured boosts -- additive, capped, never destructive.
            if wanted_types and set(d.test_type) & wanted_types:
                score += 0.15
            if wanted_level and any(
                wanted_level in jl.lower() for jl in d.job_levels
            ):
                score += 0.08

            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(1, min(k, 10))]

        # Coverage guarantee: if the user explicitly asked for certain test
        # types, make sure at least one of each requested type is present in
        # the returned set (swap in the best-scoring doc of a missing type for
        # the weakest current pick). Pure ranking can otherwise drop a whole
        # requested category, tanking Recall@10 on multi-type needs.
        if wanted_types:
            present = {tt for _, d in top for tt in d.test_type}
            for need in wanted_types:
                if need in present:
                    continue
                best = next(
                    (sd for sd in scored if need in sd[1].test_type), None
                )
                if best and best not in top:
                    top[-1] = best
                    top.sort(key=lambda x: x[0], reverse=True)
                    present |= set(best[1].test_type)

        return [self._raw[d.idx] | {"_score": round(s, 4)} for s, d in top]

    def get_by_name(self, name: str) -> dict | None:
        name_l = name.lower().strip()
        for d in self.docs:
            if d.name.lower() == name_l:
                return self._raw[d.idx]
        # fuzzy contains
        for d in self.docs:
            if name_l in d.name.lower() or d.name.lower() in name_l:
                return self._raw[d.idx]
        return None

    @property
    def size(self) -> int:
        return len(self.docs)
