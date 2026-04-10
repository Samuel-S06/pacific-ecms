# Pacific ECMS — Enterprise Context Management System

A modular document retrieval pipeline built in Python. It ingests documents, splits them into chunks, embeds them, retrieves relevant results, enforces permissions, reranks outputs, and assembles a final context window for LLM use.

The goal was to build a system that reflects what real enterprise retrieval looks like — not just search, but security, tradeoffs, and usability.

---

## What It Does

- **Chunking** – Splits documents using fixed, sentence-based, or semantic strategies  
- **Embedding & Search** – Converts text to vectors and retrieves relevant chunks (NumPy or FAISS)  
- **Permissions** – Role-based access control with audit logging  
- **Reranking** – Improves results using MMR or cross-encoder models  
- **Context Assembly** – Packs the best results into a clean, token-limited context  

All components are modular and can be swapped independently.

---

## Project Structure

ecms/
  chunker.py
  embedder.py
  vector_store.py
  reranker.py
  permissions.py
  context_assembler.py
  eval.py
  pipeline.py
tests/
demo.py
web_app.py

---

## How to Run

### Quick Start

git clone https://github.com/yourname/pacific-ecms  
cd pacific-ecms  
pip install numpy  
python demo.py  

### Run Tests

python -m pytest tests/ -v  

### Web Interface

pip install numpy streamlit  
streamlit run web_app.py  
# open http://localhost:8501  

---

## Optional (Better Performance)

pip install sentence-transformers faiss-cpu  

- sentence-transformers → real embeddings + better reranking  
- faiss-cpu → faster search at scale  

---

## Why This Project

I built this to understand the full retrieval pipeline end-to-end. Most demos stop at similarity search, but in practice the harder parts are:

- enforcing permissions correctly  
- balancing speed vs retrieval quality  
- avoiding redundant or low-value context  
