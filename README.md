<p align="center">
  <h1 align="center">🤖 AI Research Assistant</h1>
  <p align="center"><strong>Chat with Your Documents — No API keys, no cloud costs, 100% privacy</strong></p>
  <p align="center"><em>Upload a PDF and ask questions. Everything runs locally on your machine.</em></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-Vector_DB-0099FF?style=flat-square"/>
  <img src="https://img.shields.io/badge/HuggingFace-Models-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/100%25-Local-green?style=flat-square"/>
</p>

---

## 🧠 How It Works

This app uses **Retrieval-Augmented Generation (RAG)** — entirely offline:

```
  Your Question
       │
       ▼
  ┌──────────────────┐     Top 3 matches     ┌──────────────┐
  │  MiniLM-L6-v2    │ ──────────────────────►│  RoBERTa     │
  │  (Embeddings)    │                        │  (QA Reader) │
  │       +          │                        │              │
  │  FAISS Index     │                        │  Extracts    │
  │  (Vector Search) │                        │  exact answer│
  └──────────────────┘                        └──────────────┘
```

| Pipeline Step | Model | What It Does |
|--------------|-------|-------------|
| 🧠 **Embeddings** | `all-MiniLM-L6-v2` | Converts text into vector coordinates |
| 🔍 **Retrieval** | FAISS | Lightning-fast similarity search |
| 📖 **Reading** | `deepset/roberta-base-squad2` | Extracts precise answers from context |
| 📝 **Summarization** | `facebook/bart-large-cnn` | Condenses full document into TL;DR |

---

## ✨ Features

- 💡 **Smart Search** — understands meaning, not just keywords
- 🔒 **Privacy First** — everything runs locally via Hugging Face
- ⚡ **Fast Retrieval** — FAISS vector index for instant lookups
- 📝 **Auto-Summarization** — one-click document summary

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Nikhilchapkanade/AI-Research-Assistant.git
cd AI-Research-Assistant

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

*The app automatically loads the pre-built `faiss_index` and models.*

---

## 📁 Project Structure

```
AI-Research-Assistant/
├── app.py              # Streamlit interface + RAG pipeline
├── faiss_index/        # Pre-built vector database
├── data.pdf            # Sample document
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Interface | Streamlit |
| Orchestration | LangChain |
| Vector Store | FAISS |
| QA Model | RoBERTa (deepset/roberta-base-squad2) |
| Summarizer | BART (facebook/bart-large-cnn) |
| Embeddings | Sentence Transformers (MiniLM-L6-v2) |
