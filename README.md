# Medibot

Medibot is a Streamlit-based multimodal medical RAG assistant. It combines scraped medical content, chunked embeddings, FAISS retrieval, optional image analysis, and nearby hospital lookup to answer health-related questions.

## Features

- Chat-style medical Q&A in Streamlit
- Retrieval-augmented generation over indexed medical content
- Optional image upload for multimodal input
- Nearby hospital suggestions for urgent cases
- Ingestion pipeline for scrape, preprocess, chunk, and embed

## Project Structure

- `app.py` - Streamlit UI and chat workflow
- `setup.py` - One-shot ingestion pipeline runner
- `ingestion/` - Scraping, preprocessing, chunking, and embedding utilities
- `rag/` - Retriever, prompt builder, and response generation
- `vectorstore/` - FAISS-backed vector store helpers
- `memory/` - Chat history management
- `utils/` - Logging and location utilities
- `data/` - Processed text, embeddings, and index artifacts

## Requirements

- Python 3.10+
- A Google API key for generation and embeddings
- Optional: LocationIQ API key for nearby hospital lookup
- Optional: Ollama if you want to use the local model path configured in `config.py`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your keys:

```env
GOOGLE_API_KEY=your_google_api_key
LOCATIONIQ_API_KEY=your_locationiq_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct
```

## Build the index

Run the ingestion pipeline to scrape, preprocess, chunk, and embed the source content:

```bash
python setup.py
```

## Run the app

Start the Streamlit interface:

```bash
streamlit run app.py
```

## Notes

- If the FAISS index is missing, the app will prompt you to run the ingestion pipeline first.
- The app is not a substitute for professional medical advice.