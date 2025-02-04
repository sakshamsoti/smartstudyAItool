# DocuDive - PDF Query Application

DocuDive is a Streamlit-based web application that allows users to upload PDFs, extract text, and query the document using OpenAI's GPT models.

## Features
- Upload PDFs for processing.
- Extract text and search through PDFs using AI.
- Uses LangChain and MongoDB for efficient document retrieval.

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
docudive/
├── backend/    # Backend processing
│   ├── backend.py
│   ├── requirements.txt
├── frontend/   # Streamlit app
│   ├── app.py
│   ├── requirements.txt
└── README.md
```
