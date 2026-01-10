# 🤖 Local RAG: Chat with Your Documents

**No API keys. No cloud costs. 100% Privacy.**

## 👋 What is this?
Ever tried to find a specific answer in a 50-page syllabus or research paper? It’s a pain.

I built this **AI Research Assistant** to solve that. It allows you to upload a PDF (like a university course syllabus) and chat with it naturally. You can ask questions like *"What are the prerequisites?"* or *"Summarize Module 3,"* and it gives you instant answers based *strictly* on the document content.

Unlike most AI wrappers, this **runs entirely on your machine**. It doesn't send your data to ChatGPT or Google.

## ✨ Key Features
* **💡 Smart Search:** Uses vector embeddings to understand the *meaning* of your question, not just keyword matching.
* **🔒 Privacy First:** Everything runs locally using Hugging Face models.
* **⚡ Fast Retrieval:** Powered by FAISS for lightning-fast lookups.
* **📝 Auto-Summarization:** Can read the whole doc and write a summary for you.

## 🧠 How It Works (The "Magic")
This app utilizes a technique called **Retrieval-Augmented Generation (RAG)**. Here is the pipeline under the hood:

1.  **The "Brain" (Vector Store):** I used `all-MiniLM-L6-v2` to convert the document text into vector embeddings—essentially turning language into coordinates—and stored them in a **FAISS** index.
2.  **The "Retriever":** When you ask a question, the system searches the FAISS index for the most relevant paragraphs.
3.  **The "Reader":** It feeds those paragraphs into `deepset/roberta-base-squad2`, a Transformer model specialized in reading comprehension, to extract the exact answer.
4.  **The "Summarizer":** For general overviews, I implemented `facebook/bart-large-cnn` to distill the text into a concise abstract.

## 🛠️ Tech Stack
* **Language:** Python
* **Interface:** Streamlit
* **Orchestration:** LangChain
* **ML Models:** Hugging Face Transformers (`RoBERTa`, `BART`, `MiniLM`)

## 🚀 Getting Started

1.  **Clone the repo**
    ```bash
    git clone [https://github.com/your-username/ai-research-assistant.git](https://github.com/your-username/ai-research-assistant.git)
    cd ai-research-assistant
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```
    *The app will automatically load the pre-built `faiss_index` "brain" and models.*

## 📸 Screenshots
*(You can add a screenshot of your interface here later)*

## 🔮 What's Next?
* Adding support for users to upload their own PDFs dynamically.
* Integrating a generative LLM (like Llama-3) for more conversational replies.
* Adding chat history memory.

---
*Built with ❤️ by [Your Name]*
