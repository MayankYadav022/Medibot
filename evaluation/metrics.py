"""
Robust RAG Evaluation Metrics (LLM-as-judge, stabilized)
"""
import re
import requests
import google.generativeai as genai
from config import GOOGLE_API_KEY, GEMINI_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

_gemini_model = None
if GOOGLE_API_KEY:
    _gemini_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 10,
        },
    )
# ── Safe scoring parser ───────────────────────────────────────────────────────
def _parse_score(text: str) -> float:
    nums = re.findall(r"\d+\.?\d*", text)
    if not nums:
        return 0.5

    val = float(nums[0])

    # Accept either normalized scores (0.0-1.0) or 0-10 judge scores.
    if val <= 1.0:
        return max(0.0, val)

    return max(0.0, min(val / 10.0, 1.0))


def _score_with_gemini(prompt: str) -> float | None:
    try:
        if _gemini_model is None:
            return None

        res = _gemini_model.generate_content(
            prompt,
            request_options={"timeout": 20},
        )
        text = (res.text or "").strip()
        if not text:
            return None
        return _parse_score(text)
    except Exception:
        return None


def _score_with_ollama(prompt: str) -> float:
    try:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": 10,
            },
        }
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        text = (data.get("response") or "").strip()
        if not text:
            return 0.5
        return _parse_score(text)
    except Exception:
        return 0.5


def _score(prompt: str) -> float:
    gemini_score = _score_with_gemini(prompt)
    if gemini_score is not None:
        return gemini_score

    return _score_with_ollama(prompt)


# ── Build context safely ──────────────────────────────────────────────────────
def _build_context(docs: list[str], max_chars: int = 2000) -> str:
    context = ""
    for d in docs:
        if len(context) + len(d) > max_chars:
            break
        context += d + "\n\n"
    return context.strip()


# ── Metrics ───────────────────────────────────────────────────────────────────
def relevance_score(query: str, response: str) -> float:
    prompt = f"""
You are a strict medical evaluator.

Task: Score how relevant the answer is to the question.

Rules:
- Return a single score between 0 and 10.
- 10 = perfectly answers the question
- 5 = partially relevant
- 0 = completely unrelated

Return ONLY a number from 0 to 10.

Question: {query}
Answer: {response}
Score:
"""
    return _score(prompt)


def grounding_score(response: str, docs: list[str]) -> float:
    context = _build_context(docs)

    prompt = f"""
You are a strict medical evaluator.

Task: Score how well the answer is supported by the documents.

Rules:
- Return a single score between 0 and 10.
- 10 = fully supported by documents
- 5 = partially supported
- 0 = not supported

Return ONLY a number from 0 to 10.

Documents:
{context}

Answer:
{response}

Score:
"""
    return _score(prompt)

def hallucination_score(response: str, docs: list[str]) -> float:
    context = _build_context(docs)

    prompt = f"""
You are a strict medical evaluator.
Task: Detect hallucination.
Rules:
- Return a single score between 0 and 10.
- 10 = no hallucination (fully consistent)
- 5 = minor unsupported claims
- 0 = major hallucination

Return ONLY a number from 0 to 10.
Documents:{context}
Answer:{response}

Score:
"""
    return _score(prompt)

def evaluate(query: str, response: str, docs: list[str]) -> dict:
    return {
        "relevance": relevance_score(query, response),
        "grounding": grounding_score(response, docs),
        "no_hallucination": hallucination_score(response, docs),
    }