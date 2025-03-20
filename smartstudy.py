import os
import time
import re
import json
import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chat_models import ChatOpenAI  # Reverting back to ChatOpenAI
from pymongo import MongoClient
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "docudive"
COLLECTION_NAME = "pdf_embeddings"

# Connect to MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
pdf_collection = db[COLLECTION_NAME]

os.environ["OPENAI_API_KEY"] = //insert your openAI key here in ""

PDF_FOLDER = "pdfs"
UPLOADED_PDF_FOLDER = "uploaded_pdfs"

# Setting wide layout
st.set_page_config(page_title="DocuDive", layout="wide")

# Function to clean text
def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Extract images using PyMuPDF
def extract_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_data = base_image["image"]
            images.append(image_data)
    return images

# Extract tables using pdfplumber
def extract_tables_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        tables = []
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                tables.append(pd.DataFrame(table[1:], columns=table[0]))  # Convert to DataFrame
    return tables

# Load and Prepare PDF Data (text, images, tables)
@st.cache_resource
def load_and_prepare_data(pdf_path: str, chunk_size: int = 300, chunk_overlap: int = 100):
    pdf_name = os.path.basename(pdf_path)
    existing_pdf = pdf_collection.find_one({"pdf_name": pdf_name})
    
    images = []  # Initialize images to an empty list
    tables = []  # Initialize tables to an empty list
    
    if existing_pdf:
        documents = json.loads(existing_pdf["documents"])
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_texts(documents, embeddings, persist_directory="chroma_db")
    else:
        st.info(f"🔄 Processing {pdf_name} for the first time.")
        
        # Extract text using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Extract images using PyMuPDF
        images = extract_images_from_pdf(pdf_path)
        
        # Extract tables using pdfplumber
        tables = extract_tables_from_pdf(pdf_path)
        
        # Clean text data
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_documents = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(split_documents, embeddings, persist_directory="chroma_db")

        document_texts = [doc.page_content for doc in split_documents]
        pdf_collection.insert_one({
            "pdf_name": pdf_name,
            "documents": json.dumps(document_texts),
            "metadata": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        })
        st.success(f"✅ PDF processed and saved to MongoDB.")
    
    return vector_store, images, tables

# Main Streamlit App
def main():
    st.title("📄 SmartStudy: PDF Query App with Streaming Responses")
    st.write("Select or upload a PDF and ask your questions.")

    # Sidebar for PDF selection and upload
    with st.sidebar:
        st.subheader("PDF Chat Settings")
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
        uploaded_pdfs = [f for f in os.listdir(UPLOADED_PDF_FOLDER) if f.endswith('.pdf')]
        all_pdfs = pdf_files + uploaded_pdfs

        if all_pdfs:
            selected_pdf = st.selectbox("Choose a PDF to Search:", all_pdfs)
        else:
            st.error("No PDFs available. Please upload one.")
            selected_pdf = None

        # PDF upload
        uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
        if uploaded_file:
            os.makedirs(UPLOADED_PDF_FOLDER, exist_ok=True)
            uploaded_file_path = os.path.join(UPLOADED_PDF_FOLDER, uploaded_file.name)
            with open(uploaded_file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.sidebar.success(f"Uploaded {uploaded_file.name}")

    # Ensure selected_pdf is properly referenced
    if selected_pdf:
        if selected_pdf in pdf_files:
            pdf_path = os.path.join(PDF_FOLDER, selected_pdf)
        elif selected_pdf in uploaded_pdfs:
            pdf_path = os.path.join(UPLOADED_PDF_FOLDER, selected_pdf)
        else:
            st.error("Selected PDF not found in either directory.")
            st.stop()

        vector_store, images, tables = load_and_prepare_data(pdf_path)

    # Conversation History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat Area
    st.subheader("🧠 Start Asking Questions")
    for msg in st.session_state.messages:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "ai"):
            st.markdown(msg.content)

    user_input = st.chat_input("Type your question here...")
    if user_input:
        # Append user message to history
        user_message = HumanMessage(content=user_input)
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_message.content)

        if selected_pdf:
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # Retrieve relevant documents
            relevant_docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Conversation history
            conversation_history = "\n".join(
                [f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in st.session_state.messages]
            )

            if not context.strip():
                st.chat_message("ai").markdown("I could not find any relevant information in the document.")
            else:
                prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context:
{context}

The conversation so far:
{history}

The user asked:
{query}

If no relevant information is found, respond: "I could not find any relevant information in the document. Do not include any prefixes like Assistant in your response."
""")
                prompt = prompt_template.format(context=context, history=conversation_history, query=user_input)

                human_message = HumanMessage(content=prompt)

                llm = ChatOpenAI(model="gpt-4", streaming=True)
                with st.chat_message("ai"):
                    response_placeholder = st.empty()
                    ai_message = AIMessage(content="")
                    for chunk in llm.stream([human_message]):
                        if hasattr(chunk, "content"):
                            ai_message.content += chunk.content
                            response_placeholder.markdown(ai_message.content)

                st.session_state.messages.append(ai_message)

if __name__ == "__main__":
    main()
