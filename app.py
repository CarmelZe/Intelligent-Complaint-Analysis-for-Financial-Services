# app.py

import streamlit as st
from task3_rag_pipeline import RAGPipeline

# Initialize RAG pipeline
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

rag = load_pipeline()

# App UI
st.set_page_config(page_title="CrediTrust Complaint Assistant", layout="centered")
st.title("📊 CrediTrust Complaint Answering Assistant")
st.markdown("Ask any question related to customer complaints (e.g., *Why are people unhappy with BNPL?*)")

# User input
question = st.text_input("💬 Your question:", placeholder="Type your question here...")

# Buttons
col1, col2 = st.columns([1, 1])
submit_clicked = col1.button("🔍 Ask")
clear_clicked = col2.button("🧹 Clear")

# Session state to preserve results
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "sources" not in st.session_state:
    st.session_state.sources = []

# Clear button logic
if clear_clicked:
    st.session_state.answer = ""
    st.session_state.sources = []
    st.experimental_rerun()

# Submit button logic
if submit_clicked and question.strip():
    with st.spinner("Retrieving complaints and generating answer..."):
        answer, chunks = rag.run_pipeline(question)
        st.session_state.answer = str(answer).strip()
        st.session_state.sources = chunks

# Display answer
if st.session_state.answer:
    st.markdown("### 🤖 Generated Answer:")
    st.success(st.session_state.answer)

# Display sources
if st.session_state.sources:
    st.markdown("### 📚 Retrieved Complaint Excerpts:")
    for idx, chunk in enumerate(st.session_state.sources):
        st.markdown(f"""
        **{idx + 1}. {chunk['product']}**  
        *Complaint ID:* {chunk['complaint_id']}  
        {chunk['chunk_text']}
        """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for Task 4 | *Note: This is a simulated demo using fake data.*")
