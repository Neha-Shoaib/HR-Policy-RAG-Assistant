# 📚 HR Policy Assistant — RAG

An AI-powered HR Policy Assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded HR policy PDFs.

The application extracts text from a PDF, divides it into chunks, converts the chunks into semantic embeddings using Sentence Transformers, stores them in a FAISS vector index, retrieves relevant policy sections, and sends the retrieved context to an LLM for a grounded answer.

## 🚀 Features

- Upload HR policy PDFs
- Extract PDF text using PyMuPDF
- Semantic text chunking
- Sentence Transformer embeddings
- FAISS vector similarity search
- RAG-based question answering
- Grok support through the xAI API
- OpenAI GPT-OSS 120B support
- Page-aware source citations
- Retrieved source inspection
- Chat history
- No external database required
- Deployable through Streamlit Community Cloud

## 🧠 RAG Pipeline

```text
PDF
 ↓
PyMuPDF
 ↓
Text extraction
 ↓
Text chunking
 ↓
Sentence Transformers
 ↓
Embeddings
 ↓
FAISS
 ↓
Similarity Search
 ↓
Relevant Policy Chunks
 ↓
Grok / GPT-OSS 120B
 ↓
Grounded Answer + Sources
