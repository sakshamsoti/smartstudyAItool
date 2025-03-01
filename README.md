SmartStudy - Multi-Agent PDF Query Application:

SmartStudy is a multi-agent Streamlit-based web application that allows users to upload PDFs, extract text, and query documents using OpenAI's GPT models. The application now leverages LangGraph to implement a multi-agent architecture, improving modularity, efficiency, and scalability.

Features
✅ Multi-Agent Architecture using LangGraph for modular and parallel processing.
✅ PDF Upload & Processing – Extracts text, images, and tables from PDFs.
✅ Efficient Document Retrieval – Uses LangChain with MongoDB & Chroma for vector search.
✅ Conversational AI – Enables users to query documents and receive AI-generated responses.

How It Works
The system is built using a multi-agent pipeline, where specialized agents handle different responsibilities:

Document Processor Agent – Extracts text, images, and tables from PDFs.
Vector Storage Agent – Chunks text, generates embeddings, and stores data in Chroma.
Chat Agent – Handles user queries by retrieving relevant content and generating AI responses.
Orchestrator (LangGraph) – Manages the workflow between agents for efficient execution.
Installation

# Install dependencies
pip install -r backend/requirements.txt

pip install -r frontend/requirements.txt

# Run the backend (if needed in future expansion)
python backend/backend.py

# Run the Streamlit app
streamlit run frontend/app.py
Project Structure

SmartStudyPDFQueryApp/
├── backend/    # Backend processing with LangGraph multi-agent architecture
│   ├── backend.py
│   ├── requirements.txt
├── frontend/   # Streamlit-based user interface
│   ├── app.py
│   ├── requirements.txt
└── README.md   # Project documentation
Multi-Agent Workflow
The system uses LangGraph to define a structured workflow:

graph TD
    A[User Uploads PDF] -->|Process Document| B[Document Processor Agent]
    B -->|Extract Text, Images, Tables| C[Vector Storage Agent]
    C -->|Store & Retrieve Embeddings| D[Chat Agent]
    D -->|Generate AI Response| E[User Gets Answer]
Each agent operates independently, allowing efficient task delegation and modular improvements.

Usage
Upload a PDF through the Streamlit UI.
The Document Processor Agent extracts text, images, and tables.
The Vector Storage Agent processes text into chunks and stores embeddings.
The Chat Agent retrieves relevant content and formulates a response using GPT.
The orchestrator ensures smooth execution and conversation tracking.
