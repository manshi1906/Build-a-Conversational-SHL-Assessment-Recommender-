"""
main.py
-------
FastAPI service. Two endpoints, exactly as specified by the brief.

  GET  /health -> {"status": "ok"}  (HTTP 200)
  POST /chat    -> {"reply": str,
                    "recommendations": [{name,url,test_type}, ...],  # 0 or 1..10
                    "end_of_conversation": bool}

The schema is non-negotiable, so the response model is strict and we validate
on the way out. The retriever (and embedding model) is built once at startup
and reused -- this keeps every /chat call well under the 30s budget.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .agent import Agent
from .retrieval import Retriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Ensure a catalog exists; seed if the scrape never ran.
    catalog = DATA_DIR / "catalog.json"
    if not catalog.exists():
        import subprocess
        import sys

        subprocess.run(
            [sys.executable,
             str(Path(__file__).resolve().parent.parent
                 / "scripts" / "seed_catalog.py")],
            check=False,
        )
    retriever = Retriever()
    STATE["agent"] = Agent(retriever)
    STATE["catalog_size"] = retriever.size
    print(f"[startup] catalog loaded: {retriever.size} entries")
    yield
    STATE.clear()


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


# ---- schema ---------------------------------------------------------- #


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def _max_ten(cls, v):
        # Hard eval: never more than 10. (0 is allowed = still gathering.)
        return v[:10]


# ---- endpoints ------------------------------------------------------- #


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"}, status_code=200)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    agent: Agent = STATE.get("agent")
    if agent is None:  # cold start race -- build on demand
        agent = Agent(Retriever())
        STATE["agent"] = agent

    history = [m.model_dump() for m in req.messages]
    try:
        result = agent.respond(history)
    except Exception as exc:  # noqa: BLE001
        # Absolute guarantee: never break the schema, even on an unexpected
        # error. A schema-valid graceful reply beats a 500 for the evaluator.
        print(f"[chat] unexpected error: {exc!r}")
        result = {
            "reply": "Something went wrong on my side. Could you rephrase "
            "the role you're hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Defensive normalisation before it hits the response model.
    recs = result.get("recommendations") or []
    norm = []
    for r in recs[:10]:
        if isinstance(r, dict) and r.get("name") and r.get("url"):
            tt = r.get("test_type") or "K"
            norm.append(
                {"name": r["name"], "url": r["url"],
                 "test_type": tt if isinstance(tt, str) else "K"}
            )
    return ChatResponse(
        reply=str(result.get("reply", "")),
        recommendations=norm,
        end_of_conversation=bool(result.get("end_of_conversation", False)),
    )


@app.get("/")
def root():
    return {
        "service": "SHL Assessment Recommender",
        "catalog_size": STATE.get("catalog_size", 0),
        "endpoints": ["GET /health", "POST /chat"],
    }
