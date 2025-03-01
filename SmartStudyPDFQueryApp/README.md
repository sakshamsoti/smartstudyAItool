# SmartStudy - PDF Query Application

SmartStudy is a multi-agent Streamlit-based web application that allows users to upload PDFs, extract text, and query documents using OpenAI's GPT models. The application now leverages LangGraph to implement a multi-agent architecture, improving modularity, efficiency, and scalability.

## Features
✅ Multi-Agent Architecture using LangGraph for modular and parallel processing.
✅ PDF Upload & Processing – Extracts text, images, and tables from PDFs.
✅ Efficient Document Retrieval – Uses LangChain with MongoDB & Chroma for vector search.
✅ Conversational AI – Enables users to query documents and receive AI-generated responses.

## Installation
```sh
# Install dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Run the backend (if needed in future expansion)
python backend/backend.py

# Run the Streamlit app
streamlit run frontend/app.py
```

## Project Structure
```
SmartStudyPDFQueryApp/
├── backend/    # Backend processing with LangGraph multi-agent architecture
│   ├── backend.py
│   ├── requirements.txt
├── frontend/   # Streamlit-based user interface
│   ├── app.py
│   ├── requirements.txt
└── README.md   # Project documentation
```
## Multi-Agent Workflow

```
graph TD
    A[User Uploads PDF] -->|Process Document| B[Document Processor Agent]
    B -->|Extract Text, Images, Tables| C[Vector Storage Agent]
    C -->|Store & Retrieve Embeddings| D[Chat Agent]
    D -->|Generate AI Response| E[User Gets Answer]
```
