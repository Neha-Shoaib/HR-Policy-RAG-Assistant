# 📚 HR Policy Assistant

An AI-powered HR Policy Assistant built using Retrieval-Augmented Generation (RAG).

The application allows a user to upload an HR policy PDF and ask questions about the document. The system retrieves the most relevant sections of the policy using semantic search and FAISS, then uses Groq's `openai/gpt-oss-120b` model to generate a grounded answer.

---

## 🚀 Features

- Upload HR policy PDF
- Extract text using PyMuPDF
- Preserve page numbers
- Automatic text chunking
- Semantic embeddings using Sentence Transformers
- FAISS vector similarity search
- Retrieval-Augmented Generation (RAG)
- Groq API integration
- OpenAI GPT-OSS 120B model through Groq
- Grounded answers
- Page-level source information
- Similarity scores
- Chat interface
- Chat history
- No external vector database required
- Streamlit deployment ready

---

# 🧠 How RAG Works

The application follows this pipeline:

```text
HR Policy PDF
      ↓
   PyMuPDF
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Sentence Transformers
      ↓
Embeddings
      ↓
FAISS Vector Index
      ↓
User Question
      ↓
Question Embedding
      ↓
Similarity Search
      ↓
Relevant Policy Chunks
      ↓
Groq
      ↓
openai/gpt-oss-120b
      ↓
Grounded Answer
      ↓
Source Pages
