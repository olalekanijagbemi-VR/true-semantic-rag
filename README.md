# Professional RAG System

**Live Demo:** https://true-semantic-rag-53sy8qr4z4oeaa88uqx6q8.streamlit.app/

## Tech Stack
| Component | Technology |
|-----------|------------|
| Vector Database | FAISS |
| Embeddings | HuggingFace (all-mpnet-base-v2) |
| LLM | Groq (llama-3.3-70b-versatile) |
| Framework | LangChain |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

## Features
- ✅ Semantic search with cosine similarity (NOT keyword matching)
- ✅ Document summarization
- ✅ Hallucination guardrails (confidence threshold 0.15)
- ✅ Live deployment - test with your own PDFs

## Quick Start
```bash
git clone https://github.com/olalekanijagbemi-VR/true-semantic-rag
cd true-semantic-rag
pip install -r requirements.txt
streamlit run app.py