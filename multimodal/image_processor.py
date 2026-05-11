"""
Send an image to Gemini Vision and get back a medical text description.
`image` can be a PIL.Image or a file path string.
"""
import google.generativeai as genai
from PIL import Image
from config import GOOGLE_API_KEY, GEMINI_VISION_MODEL

# Initialize Gemini with API key if available
_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        _model = genai.GenerativeModel(GEMINI_VISION_MODEL)
    except Exception as init_error:
        # Log but don't crash - image processing will fail gracefully at runtime
        print(f"[Warning] Could not initialize Gemini Vision model: {init_error}")
        _model = None

VISION_PROMPT = (
    "You are a medical image analysis assistant. "
    "Describe this medical image in detail: identify any visible symptoms, "
    "anatomical structures, lesions, rashes, abnormalities, or medical conditions. "
    "Be factual and clinically descriptive."
)
def process_image(image) -> str:
    """Process image with Gemini Vision if available, else return graceful message."""
    if _model is None:
        return "[Image upload not available - Gemini Vision model unavailable. Proceeding with text analysis only.]"
    
    try:
        if isinstance(image, str):
            image = Image.open(image)
        response = _model.generate_content(
            [VISION_PROMPT, image],
            request_options={"timeout": 20},
        )
        text = (response.text or "").strip()
        if not text:
            return "[Image processed but no description generated]"
        return text
    except Exception as e:
        # Graceful fallback: Gemini vision not available, skip image processing
        error_msg = str(e).lower()
        if "404" in error_msg or "not found" in error_msg or "not supported" in error_msg:
            return "[Image upload not available - model not found. Proceeding with text analysis only.]"
        elif "quota" in error_msg or "resource_exhausted" in error_msg:
            return "[Image quota exhausted. Proceeding with text analysis only.]"
        else:
            return f"[Image processing skipped: {type(e).__name__}. Continuing with text analysis only.]"