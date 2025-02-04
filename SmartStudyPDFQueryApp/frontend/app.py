import os
import streamlit as st
from backend.backend import process_query

PDF_FOLDER = "pdfs"

# Streamlit App
st.set_page_config(page_title="DocuDive", layout="wide")
st.title("📄 DocuDive: PDF Query App")
st.write("Select or upload a PDF and ask your questions.")

with st.sidebar:
    st.subheader("PDF Chat Settings")
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    uploaded_pdfs = [f for f in os.listdir("uploaded_pdfs") if f.endswith('.pdf')]
    all_pdfs = pdf_files + uploaded_pdfs

    if all_pdfs:
        selected_pdf = st.selectbox("Choose a PDF to Search:", all_pdfs)
    else:
        st.error("No PDFs available. Please upload one.")
        selected_pdf = None

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if uploaded_file:
        os.makedirs("uploaded_pdfs", exist_ok=True)
        with open(f"uploaded_pdfs/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())
        st.sidebar.success(f"Uploaded {uploaded_file.name}")

st.subheader("🧠 Start Asking Questions")
user_input = st.chat_input("Type your question here...")

if user_input and selected_pdf:
    response = process_query(user_input, selected_pdf)
    st.chat_message("ai").markdown(response)
