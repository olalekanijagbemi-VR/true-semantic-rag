# True Semantic RAG System

**Production RAG with pure vector similarity search - NO keyword matching, NO BM25**

## Demo Video
[Add your Loom/Walkthrough link here]

## Features
- ✅ **Pure Semantic Search** - Cosine similarity on vector embeddings
- ✅ **No Hallucinations** - Confidence thresholding (0.35+ relevance required)
- ✅ **Production Ready** - Groq inference (300 tokens/sec)
- ✅ **Zero Cost** - Free tier Groq API

## Tech Stack
| Component | Technology |
|-----------|------------|
| Embeddings | sentence-transformers/all-mpnet-base-v2 (768 dims) |
| Vector DB | ChromaDB with cosine similarity |
| LLM | Groq (llama-3.3-70b-versatile) |
| UI | Streamlit |
| Framework | LangChain |

## Quick Start

```bash
git clone https://github.com/olalekanijagbemi-VR/true-semantic-rag.git
cd true-semantic-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py