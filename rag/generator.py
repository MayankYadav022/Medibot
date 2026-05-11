import requests
import google.generativeai as genai
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, GOOGLE_API_KEY, GEMINI_MODEL

# Configure Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


def _build_ollama_prompt(prompt: str) -> str:
    return (
        "You are a concise medical assistant. Answer clearly, accurately, and briefly. "
        "If the context is insufficient, say so instead of guessing.\n\n"
        f"{prompt}"
    )


def _generate_with_gemini(prompt: str) -> str:
    """Try to generate with Gemini. Returns error sentinel if quota/unavailable."""
    try:
        if not GOOGLE_API_KEY:
            return "__GEMINI_NOT_CONFIGURED__"
        
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            request_options={"timeout": 30},
        )
        text = response.text.strip() if response.text else ""
        if not text:
            return "__GEMINI_NO_RESPONSE__"
        return text
    except Exception as e:
        error_msg = str(e).lower()
        # Check for quota errors
        if "429" in error_msg or "quota" in error_msg or "resource_exhausted" in error_msg:
            return "__GEMINI_QUOTA_EXHAUSTED__"
        # Check for model not found errors
        elif "404" in error_msg or "not found" in error_msg:
            return "__GEMINI_MODEL_NOT_FOUND__"
        # Other errors (connection, timeout, etc)
        else:
            return "__GEMINI_ERROR__"


def _generate_with_ollama(prompt: str) -> str:
    """Generate with local Ollama model."""
    try:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": _build_ollama_prompt(prompt),
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 512,
            },
        }

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        text = (data.get("response") or "").strip()
        if not text:
            return "I'm unable to generate a response right now."

        return text
    except Exception as e:
        error_text = str(e)
        if "connection" in error_text.lower() or "refused" in error_text.lower():
            return "__OLLAMA_UNAVAILABLE__"
        else:
            return "__OLLAMA_ERROR__"


def generate(prompt: str) -> str:
    """
    Try Gemini first. If it hits quota limits, fall back to Ollama.
    If Ollama also fails, return error sentinel.
    """
    # Attempt Gemini first
    gemini_response = _generate_with_gemini(prompt)
    
    # Check if Gemini succeeded (no error sentinel)
    if not gemini_response.startswith("__"):
        return gemini_response
    
    # Gemini failed or unavailable - fall back to Ollama
    print(f"[Fallback to Ollama] Gemini returned: {gemini_response}")
    ollama_response = _generate_with_ollama(prompt)
    
    # Check if Ollama succeeded
    if not ollama_response.startswith("__"):
        return ollama_response
    
    # Both failed - return appropriate error
    if "__GEMINI_QUOTA_EXHAUSTED__" in gemini_response:
        return "__QUOTA_EXHAUSTED__"
    elif "__OLLAMA_UNAVAILABLE__" in ollama_response:
        return "__OLLAMA_UNAVAILABLE__"
    else:
        return "__MODEL_UNAVAILABLE__"