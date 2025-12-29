import streamlit as st
from utils import load_pdf, load_text, chunk_text
from rag import create_vector_store, retrieve, generate_answer

st.set_page_config(page_title="Mini ChatGPT with Docs", layout="wide")

st.title("📄 Mini ChatGPT – Ask Your Documents")

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        raw_text = load_pdf(uploaded_file)
    else:
        raw_text = load_text(uploaded_file)

    chunks = chunk_text(raw_text)

    with st.spinner("Creating embeddings..."):
        index, stored_chunks = create_vector_store(chunks)

    st.success("Document processed! Ask a question 👇")

    query = st.text_input("Your question")

    if query:
        with st.spinner("Thinking..."):
            relevant_chunks = retrieve(query, index, stored_chunks)
            context = "\n\n".join(relevant_chunks)
            answer = generate_answer(query, context)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved context"):
            st.write(context)
