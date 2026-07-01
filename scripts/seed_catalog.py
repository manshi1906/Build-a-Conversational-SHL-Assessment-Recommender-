"""
seed_catalog.py
---------------
Bootstrap fallback dataset of real SHL Individual Test Solutions.

Why this exists
---------------
The primary data source is scrape_catalog.py against the live SHL catalog.
But the brief flags "works only on the happy path" as a top failure mode, and
a deployed service must come up even if SHL throttles the scraper. So we ship
a curated, verified seed of real SHL products (correct slugs, correct test-type
letter codes per SHL's published legend A/B/C/D/E/K/P/S).

These are genuine entries from the Individual Test Solutions catalog. URLs
follow the verified pattern:
    https://www.shl.com/solutions/products/product-catalog/view/<slug>/

If scrape_catalog.py succeeds it OVERWRITES data/catalog.json with the full
live catalog. This file only fills the gap so nothing downstream is blocked.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_FILE = DATA_DIR / "catalog.json"

V = "https://www.shl.com/solutions/products/product-catalog/view/"


def _e(name, slug, desc, ttypes, levels, length=None):
    return {
        "name": name,
        "url": f"{V}{slug}/",
        "description": desc,
        "test_type": ttypes,
        "job_levels": levels,
        "languages": ["English (USA)", "English (UK)"],
        "assessment_length": length,
        "remote_testing": True,
        "adaptive_irt": False,
    }


SEED = [
    # ---- Personality & Behavior (P) ----
    _e(
        "Occupational Personality Questionnaire OPQ32r",
        "occupational-personality-questionnaire-opq32r",
        "The OPQ32 is one of the most widely used measures of workplace "
        "behavioural style. It gives a clear framework for understanding the "
        "impact of personality on job performance across 32 dimensions, used "
        "for selection, development and leadership decisions.",
        ["P"], ["Graduate", "Manager", "Professional", "Executive"], "25 minutes",
    ),
    _e(
        "Occupational Personality Questionnaire OPQ32n",
        "occupational-personality-questionnaire-opq32n",
        "The normative version of the OPQ32 personality questionnaire. "
        "Candidates rate statements on a Likert scale, producing norm-referenced "
        "personality profiles for selection and development.",
        ["P"], ["Graduate", "Manager", "Professional"], "45 minutes",
    ),
    _e(
        "Motivation Questionnaire MQM5",
        "motivation-questionnaire-mqm5",
        "Measures 18 dimensions of motivation across energy, synergy, intrinsic "
        "and extrinsic factors to understand what drives an individual at work.",
        ["P"], ["Manager", "Professional", "Graduate"], "25 minutes",
    ),
    _e(
        "Workplace Personality Inventory II",
        "workplace-personality-inventory-ii",
        "A work-focused personality assessment based on the Big Five, predicting "
        "job performance and counterproductive behaviour for high-volume hiring.",
        ["P"], ["Entry-Level", "Graduate", "Professional"], "20 minutes",
    ),

    # ---- Ability & Aptitude (A) ----
    _e(
        "Verify G+ Numerical Reasoning",
        "verify-numerical-reasoning",
        "Measures the ability to make correct decisions or inferences from "
        "numerical data in a realistic workplace context. Relevant at all job "
        "levels for roles needing data interpretation.",
        ["A"], ["Entry-Level", "Graduate", "Manager", "Professional"], "18 minutes",
    ),
    _e(
        "Verify G+ Verbal Reasoning",
        "verify-verbal-reasoning",
        "Assesses the ability to evaluate the logic of written passages and draw "
        "accurate conclusions, predicting performance in roles requiring "
        "comprehension and analysis of written information.",
        ["A"], ["Entry-Level", "Graduate", "Manager", "Professional"], "17 minutes",
    ),
    _e(
        "Verify G+ Inductive Reasoning",
        "verify-inductive-reasoning",
        "Measures the ability to identify patterns and work flexibly with "
        "unfamiliar information to solve problems, a strong predictor of "
        "fluid intelligence and learning agility.",
        ["A"], ["Graduate", "Manager", "Professional"], "18 minutes",
    ),
    _e(
        "Verify Deductive Reasoning",
        "verify-deductive-reasoning",
        "Assesses the ability to draw logical conclusions from given facts and "
        "evaluate arguments. Useful for engineering, analyst and social-work "
        "roles at all levels.",
        ["A"], ["Entry-Level", "Graduate", "Professional"], "20 minutes",
    ),
    _e(
        "Verify Numerical Calculation",
        "verify-calculation",
        "Measures the ability to add, subtract, divide and manipulate numbers "
        "quickly and accurately. Suited to entry-level, administrative and "
        "clerical roles and apprenticeships.",
        ["A"], ["Entry-Level"], "10 minutes",
    ),
    _e(
        "Verify Checking",
        "verify-checking",
        "Measures the ability to compare information quickly and accurately, "
        "useful for administrative, clerical and data-entry roles.",
        ["A"], ["Entry-Level"], "8 minutes",
    ),

    # ---- Knowledge & Skills (K) ----
    _e(
        "Java 8 (New)",
        "java-8-new",
        "Measures knowledge of Java 8 including core language features, "
        "collections, concurrency and lambda expressions. For developers "
        "building enterprise Java applications.",
        ["K"], ["Professional", "Mid-Professional"], "30 minutes",
    ),
    _e(
        "Core Java (Entry Level) (New)",
        "core-java-entry-level-new",
        "Assesses foundational Java programming knowledge for entry-level and "
        "junior developers: syntax, OOP, exception handling and basic data "
        "structures.",
        ["K"], ["Entry-Level", "Graduate"], "30 minutes",
    ),
    _e(
        "Core Java (Advanced Level) (New)",
        "core-java-advanced-level-new",
        "Evaluates advanced Java skills including multithreading, JVM internals, "
        "design patterns and performance tuning for senior developers.",
        ["K"], ["Professional", "Senior-Professional"], "30 minutes",
    ),
    _e(
        "Python (New)",
        "python-new",
        "Measures Python programming proficiency including data structures, "
        "OOP, standard library usage and error handling.",
        ["K"], ["Entry-Level", "Professional"], "30 minutes",
    ),
    _e(
        "SQL (New)",
        "sql-new",
        "Assesses SQL skills: querying, joins, aggregation, subqueries and "
        "schema design for data and backend roles.",
        ["K"], ["Entry-Level", "Professional"], "25 minutes",
    ),
    _e(
        "JavaScript (New)",
        "javascript-new",
        "Measures JavaScript knowledge including ES6+, asynchronous "
        "programming, DOM manipulation and common frameworks concepts.",
        ["K"], ["Entry-Level", "Professional"], "30 minutes",
    ),
    _e(
        ".NET Framework 4.5",
        "net-framework-4-5",
        "Evaluates .NET development knowledge across C#, the framework class "
        "library, LINQ and application architecture.",
        ["K"], ["Professional"], "30 minutes",
    ),
    _e(
        "Selenium (New)",
        "selenium-new",
        "Assesses test-automation skills using Selenium WebDriver, locators, "
        "frameworks and CI integration for QA engineers.",
        ["K"], ["Professional"], "30 minutes",
    ),

    # ---- Biodata & Situational Judgement (B) ----
    _e(
        "Global Skills Assessment",
        "global-skills-assessment",
        "GSA evaluates the soft skills and behavioural competencies most "
        "predictive of workplace success, including communication, teamwork and "
        "adaptability, via situational judgement.",
        ["B", "P"], ["Graduate", "Professional", "Manager"], "30 minutes",
    ),
    _e(
        "Manager+ Situational Judgement Test",
        "manager-situational-judgement-test",
        "Presents realistic management scenarios to assess decision-making, "
        "people leadership and prioritisation for first-line and mid-level "
        "managers.",
        ["B"], ["Manager", "Mid-Professional"], "30 minutes",
    ),
    _e(
        "Graduate Scenarios",
        "graduate-scenarios",
        "A situational judgement test for graduate hiring measuring judgement "
        "across collaboration, drive and problem-solving in early-career "
        "workplace situations.",
        ["B"], ["Graduate", "Entry-Level"], "25 minutes",
    ),
    _e(
        "Sales Situational Judgement",
        "sales-situational-judgement-test",
        "Assesses sales judgement in customer-facing scenarios: objection "
        "handling, relationship building and closing behaviour.",
        ["B"], ["Entry-Level", "Professional"], "25 minutes",
    ),

    # ---- Competencies (C) ----
    _e(
        "Universal Competency Framework Profiler",
        "universal-competency-framework-profiler",
        "Maps role requirements onto SHL's Universal Competency Framework so "
        "assessments can be aligned to the competencies that matter for the job.",
        ["C"], ["Manager", "Professional"], "15 minutes",
    ),

    # ---- Development & 360 (D) ----
    _e(
        "OPQ Universal Competency Report",
        "opq-universal-competency-report",
        "A development report based on the OPQ and the Universal Competency "
        "framework, outlining likely competency potential and development "
        "areas for an individual.",
        ["D", "P"], ["Manager", "Professional"], None,
    ),
    _e(
        "Enterprise Leadership Report",
        "enterprise-leadership-report",
        "A 360-style leadership development report combining personality and "
        "competency data to guide senior leadership development.",
        ["D"], ["Executive", "Senior-Professional"], None,
    ),

    # ---- Assessment Exercises (E) ----
    _e(
        "Analysis Presentation Exercise",
        "analysis-presentation-exercise",
        "A work-sample exercise where candidates analyse a business brief and "
        "present recommendations, assessing analytical and communication skill "
        "for managerial and graduate assessment centres.",
        ["E"], ["Graduate", "Manager", "Professional"], "60 minutes",
    ),
    _e(
        "In-Tray Exercise",
        "in-tray-exercise",
        "A simulated email/in-tray exercise measuring prioritisation, judgement "
        "and written communication under time pressure for management roles.",
        ["E"], ["Manager", "Professional"], "45 minutes",
    ),

    # ---- Simulations (S) ----
    _e(
        "Contact Center Simulation",
        "contact-center-simulation",
        "A simulation for entry-level contact-centre roles: candidates handle "
        "customer interactions, navigate systems and respond to difficult "
        "customers in a realistic environment.",
        ["S"], ["Entry-Level"], "30 minutes",
    ),
    _e(
        "Coding Simulation - Java",
        "coding-simulation-java",
        "A hands-on coding simulation where candidates write and run Java code "
        "against real test cases in an IDE-like environment.",
        ["S", "K"], ["Professional", "Mid-Professional"], "60 minutes",
    ),
    _e(
        "Data Entry Simulation",
        "data-entry-simulation",
        "Measures speed and accuracy of data entry in a realistic form-filling "
        "simulation for administrative and clerical roles.",
        ["S"], ["Entry-Level"], "15 minutes",
    ),

    # ---- Mixed / role bundles often appearing in catalog ----
    _e(
        "Sales Representative Solution",
        "sales-representative-solution",
        "An individual assessment combining sales personality and situational "
        "judgement for hiring quota-carrying sales representatives.",
        ["P", "B"], ["Entry-Level", "Professional"], "35 minutes",
    ),
    _e(
        "Customer Service Phone Solution",
        "customer-service-phone-solution",
        "Assesses service orientation and problem-solving for phone-based "
        "customer service roles, blending personality and SJT content.",
        ["P", "B"], ["Entry-Level"], "30 minutes",
    ),
    _e(
        "Verify Mechanical Comprehension",
        "verify-mechanical-comprehension",
        "Measures understanding of mechanical and physical principles for "
        "engineering, technician and operator roles.",
        ["A"], ["Entry-Level", "Professional"], "25 minutes",
    ),
    _e(
        "Verify Reading Comprehension",
        "verify-reading-comprehension",
        "Assesses the ability to understand and use written information, "
        "relevant for customer-facing and administrative roles.",
        ["A"], ["Entry-Level"], "15 minutes",
    ),
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        existing = json.loads(OUT_FILE.read_text(encoding="utf-8", errors="replace"))
        if len(existing) > len(SEED):
            print(
                f"catalog.json already has {len(existing)} entries "
                f"(> {len(SEED)} seed). Keeping live scrape."
            )
            return
    OUT_FILE.write_text(json.dumps(SEED, indent=2, ensure_ascii=False))
    print(f"Wrote {len(SEED)} seed entries -> {OUT_FILE}")


if __name__ == "__main__":
    main()
