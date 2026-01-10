import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import os

# --- 1. SET UP THE CORE AI MODELS ---

st.set_page_config(page_title="AI Research Assistant")
st.title("AI Research Assistant: Chat with Your PDF 💬")

# Define file paths
DB_PATH = "faiss_index"

# --- This is the "Optimization" part ---
# We use @st.cache_resource to load the models only ONCE.
# This makes the app super fast after the first run.
@st.cache_resource
def load_models():
    """
    Loads all AI models and the vector database.
    This function is cached by Streamlit for performance.
    """
    print("Loading models... this will take a moment.")
    
    # 1. Load the "Brain" (Vector Database)
    # This is the "all-MiniLM-L6-v2" model from Ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if the FAISS index exists before loading
    if not os.path.exists(DB_PATH):
        st.error(f"Error: FAISS index not found at {DB_PATH}. Please run the ingest.py script first.")
        return None, None, None
        
    vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # 2. Load the Question-Answering (Q&A) Model
    # This is a "BERT" style model
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2"
    )
    
    # 3. Load the Summarization Model
    # Using facebook/bart-large-cnn - doesn't require SentencePiece
    summarizer_pipeline = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    
    # 4. Create the retriever
    # This will retrieve relevant text chunks from the vector database
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 most relevant chunks
    
    print("Models loaded successfully.")
    return qa_pipeline, summarizer_pipeline, retriever, vector_db.docstore

# Load the models
result = load_models()

if result[0] is not None:
    qa_pipeline, summarizer_pipeline, retriever, docstore = result
else:
    qa_pipeline = None

# --- 2. BUILD THE WEB INTERFACE ---

if qa_pipeline:  # Only show the app if the models loaded
    # Section for Summarization
    st.header("Document Summary")
    if st.button("Generate Summary"):
        with st.spinner("Generating summary... this may take a moment."):
            # Get all the text from our document "brain"
            full_text = " ".join([doc.page_content for i, doc in docstore._dict.items()])
            
            # Run the summarizer
            # We limit to 2000 chars for a fast demo
            summary = summarizer_pipeline(full_text[:2000], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            st.success(summary)

    # Section for Question-Answering
    st.header("Ask a Question")
    user_question = st.text_input("What do you want to know from the document?")

    if user_question:
        with st.spinner("Finding answer..."):
            try:
                # 1. Retrieve relevant documents from the vector database
                docs = retriever.invoke(user_question)
                
                if docs:
                    # 2. Combine the context from retrieved documents
                    context = " ".join([doc.page_content for doc in docs])
                    
                    # 3. Use the QA pipeline to answer the question based on the context
                    answer = qa_pipeline(question=user_question, context=context)
                    
                    # 4. Display the answer
                    st.success(f"**Answer:** {answer['answer']}")
                    st.info(f"**Confidence:** {answer['score']:.2%}")
                    
                    # Optional: Show the source context
                    with st.expander("View Source Context"):
                        for i, doc in enumerate(docs, 1):
                            st.write(f"**Source {i}:**")
                            st.write(doc.page_content)
                            st.write("---")
                else:
                    st.warning("Could not find relevant information in the document.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.warning("Could not find an answer. Please try rephrasing your question.")
else:
    st.warning("Could not load AI models. Make sure the 'faiss_index' folder is in the same directory as app.py.")