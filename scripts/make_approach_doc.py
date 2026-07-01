"""Generate the 2-page approach document as a clean PDF."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, HRFlowable
)

OUT = "/home/claude/shl_recommender/Approach_Document.pdf"

GREEN = HexColor("#5a9e2f")
DARK = HexColor("#1a1a1a")
GREY = HexColor("#444444")

styles = getSampleStyleSheet()

h_title = ParagraphStyle(
    "hTitle", parent=styles["Title"], fontSize=17, textColor=DARK,
    spaceAfter=2, leading=20,
)
h_sub = ParagraphStyle(
    "hSub", parent=styles["Normal"], fontSize=9.5, textColor=GREEN,
    spaceAfter=10, fontName="Helvetica-Bold",
)
h2 = ParagraphStyle(
    "h2", parent=styles["Heading2"], fontSize=11.5, textColor=GREEN,
    spaceBefore=9, spaceAfter=4, fontName="Helvetica-Bold",
)
body = ParagraphStyle(
    "body", parent=styles["Normal"], fontSize=9.3, textColor=GREY,
    leading=13, spaceAfter=5, alignment=4,
)
bullet = ParagraphStyle(
    "bullet", parent=body, leftIndent=10, spaceAfter=3,
)

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    topMargin=15 * mm, bottomMargin=14 * mm,
    leftMargin=17 * mm, rightMargin=17 * mm,
)

S = []


def P(t, st=body):
    S.append(Paragraph(t, st))


def B(items, st=bullet):
    S.append(ListFlowable(
        [ListItem(Paragraph(i, st), leftIndent=12, value="•") for i in items],
        bulletType="bullet", start="•", leftIndent=8,
    ))


P("Conversational SHL Assessment Recommender", h_title)
P("Approach Document &nbsp;|&nbsp; AI Intern Take-home", h_sub)
S.append(HRFlowable(width="100%", color=GREEN, thickness=1, spaceAfter=8))

P("Problem framing", h2)
P(
    "The task is a grounded, multi-turn recommendation problem where the agent "
    "must decide <i>per turn</i> whether to clarify, recommend, refine, "
    "compare, or refuse — over a non-deterministic conversation, with a "
    "hard schema, an 8-turn cap, and a 30s/call budget. I treated it as a "
    "two-stage pipeline (route, then execute) so the LLM handles ambiguous "
    "dialogue while deterministic Python owns every hard guarantee."
)

P("Catalog &amp; data", h2)
P(
    "A defensive scraper (<font face='Courier'>scrape_catalog.py</font>) walks "
    "only the Individual Test Solutions table (<font face='Courier'>type=1</font>), "
    "follows pagination, and enriches each product detail page for "
    "description, test-type letters (A/B/C/D/E/K/P/S), job levels and length. "
    "Every selector has a fallback because SHL markup shifts and "
    "&ldquo;happy-path only&rdquo; is a named failure mode. A verified "
    "seed catalog of real SHL products (correct slugs/URLs) ships alongside "
    "so the service boots and stays schema-valid even if a scrape is "
    "throttled — the scraper overwrites it when it succeeds."
)

P("Retrieval", h2)
P(
    "Hybrid, in-process (no external vector DB — the catalog is small enough "
    "that a single process stays well inside the latency budget and "
    "simplifies deployment):"
)
B([
    "<b>Dense</b>: MiniLM (all-MiniLM-L6-v2) cosine over a cached matrix — "
    "captures paraphrased intent (&ldquo;works with stakeholders&rdquo; → "
    "personality).",
    "<b>Sparse</b>: a compact BM25 (no dependency) — captures exact names "
    "and acronyms (OPQ32r, GSA, Java 8) that embeddings blur.",
    "<b>Structured boosts</b>: additive, capped nudges for requested "
    "test-type and job level.",
    "<b>Type-coverage guarantee</b>: if the user explicitly asked for a "
    "test type, at least one item of that type is forced into the candidate "
    "set — pure ranking otherwise drops a whole requested category and "
    "tanks Recall@10 on multi-type needs.",
])
P(
    "Scores are min-max normalised per query then fused (0.6 semantic / 0.4 "
    "lexical, tuned on the dev traces). If embeddings can't load, retrieval "
    "degrades to BM25-only rather than crashing."
)

P("Agent &amp; prompt design", h2)
P(
    "<b>Stage 1 — Router</b> (one strict-JSON LLM call): given the full "
    "history plus the catalog&rsquo;s controlled vocabulary, it accumulates "
    "constraints across the <i>whole</i> conversation and picks one action. "
    "The prompt encodes explicit action rules (never recommend on a vague "
    "turn-1; <i>refine</i> carries prior constraints forward instead of "
    "restarting; injection/off-topic → refuse) and a fixed intent→test-type "
    "map so behaviour is consistent."
)
P(
    "<b>Stage 2 — Execute</b> (deterministic, plus a grounded LLM pass): for "
    "recommend/refine the model picks and orders <i>only</i> from the "
    "retrieved candidate indices; names and URLs are taken from the catalog "
    "object, never the model — a hallucinated URL is structurally "
    "impossible. Compare answers are written strictly from the named "
    "entries&rsquo; catalog descriptions. The service is stateless: working "
    "state is rebuilt from message history every call."
)

P("Hard guarantees (in code, not prompt)", h2)
B([
    "Schema always valid — coerced and Pydantic-validated on the way out, "
    "even on unexpected errors (graceful reply beats a 500).",
    "Recommendations are always 0 or 1–10 real catalog rows.",
    "Turn-cap: at the last allowed turn the agent commits to a best-effort "
    "shortlist instead of asking again — 8 turns never exceeded.",
    "Every external dependency degrades: no LLM key → safe deterministic "
    "path; LLM returns junk → JSON repair + fallbacks.",
])

P("Evaluation", h2)
P(
    "A local replay harness (<font face='Courier'>eval/run_eval.py</font>) "
    "mirrors SHL&rsquo;s grading: a rule-based simulated user answers from "
    "persona facts, says &ldquo;no preference&rdquo; otherwise, and ends on "
    "a shortlist. It scores hard-eval schema validity, Mean Recall@10, and "
    "behaviour probes (refuse off-topic/injection, no-recommend-turn-1, "
    "compare-grounded). A second suite "
    "(<font face='Courier'>tests/test_agent_pipeline.py</font>) injects a "
    "deterministic fake LLM to prove the full state machine in isolation — "
    "7/7 passing (clarify, recommend, refine-honours-edit, compare, refuse "
    "×2, turn-cap)."
)

P("What didn&rsquo;t work / how I measured improvement", h2)
B([
    "<b>Pure semantic ranking</b> dropped requested test-type categories "
    "entirely; the <i>refine→adds personality</i> probe failed. Adding the "
    "type-coverage guarantee fixed it (probe FAIL→PASS) without hurting "
    "other traces.",
    "<b>LLM retry/backoff on an unset key</b> blew the turn timeout in "
    "degraded mode; switching to fail-fast when unconfigured kept every "
    "path inside budget.",
    "<b>Schema drift risk</b>: early on the model occasionally returned "
    ">10 or malformed recs; moving all schema enforcement into Python "
    "(not the prompt) made hard-evals pass on 100% of turns even with the "
    "LLM and embeddings fully offline.",
])

P("Stack justification &amp; AI-tool usage", h2)
P(
    "FastAPI + Pydantic (strict schema, async, trivial free-tier deploy); "
    "Groq Llama-3.3-70B (fast, generous free tier, OpenAI-compatible) with "
    "a one-env-var swap to OpenAI/Gemini; sentence-transformers + a "
    "hand-written BM25 (no heavyweight vector DB for a small catalog). "
    "AI assistance was used for scaffolding, the scraper&rsquo;s defensive "
    "selectors, and prose polish; all design decisions, the routing "
    "contract, the grounding/anti-hallucination mechanism, the "
    "type-coverage fix, and the evaluation methodology are my own and are "
    "defended above."
)

doc.build(S)
print("wrote", OUT)
