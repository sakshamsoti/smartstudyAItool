# SmartStudy: AI-Powered PDF Query App

## Overview
SmartStudy is a Streamlit-based application that allows users to upload and interact with PDFs. It utilizes AI-powered text embeddings to retrieve and answer questions based on PDF content. The application supports text extraction, image extraction, and table recognition.

## Features
- Upload and process PDFs
- Extract text using PyPDFLoader
- Extract images with PyMuPDF (fitz)
- Extract tables using pdfplumber
- Store and retrieve PDF embeddings from MongoDB
- Query PDFs using AI with OpenAI embeddings and Chroma vector database
- Stream AI-generated responses in real-time

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- MongoDB
- OpenAI API Key

### Clone the Repository
```sh
$ git clone https://github.com/yourusername/docudive.git
$ cd docudive
```

### Install Dependencies
```sh
$ pip install -r requirements/requirements.txt
```

### Set Environment Variables
Create an `.env` file and add your OpenAI API key:
```sh
OPENAI_API_KEY=your_openai_api_key_here
```
Alternatively, you can set it in your shell:
```sh
$ export OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application
1. Start MongoDB (if not running already):
   ```sh
   $ mongod --dbpath /path/to/your/db
   ```
2. Run the Streamlit app:
   ```sh
   $ streamlit run app.py
   ```

## Folder Structure
```
/docudive
│── uploaded_pdfs/    # Stores user-uploaded PDFs
│── pdfs/             # Preloaded PDFs (if any)
│── chroma_db/        # Chroma vector store directory
│── smartstudy.py            # Main Streamlit application
│── requirements/     # Folder containing dependencies
│   ├── requirements.txt
│── README.md         # This file
```

## Usage
- Select a PDF from the sidebar or upload a new one.
- Ask a question related to the document.
- The AI will retrieve relevant sections and generate a response.

## Dependencies
- `streamlit`
- `pdfplumber`
- `PyMuPDF (fitz)`
- `pandas`
- `langchain`
- `pymongo`
- `openai`


## Contributions
Pull requests and contributions are welcome! Feel free to improve the project and submit PRs.

## Contact
For questions or suggestions, contact Saksham Soti at sotisaksham@mgmail.com or open an issue in the repository.
