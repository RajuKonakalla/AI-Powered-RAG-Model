# AI-Powered RAG Model

**Secure and Efficient Document-Based Question Answering System**

---

## Table of Contents

- [About](#about)  
- [Project Overview](#project-overview)  
- [Features](#features)  
- [Skills Utilized](#skills-utilized)  
- [Technologies](#technologies)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Contributing](#contributing)  
- [License](#license)  

---

## About

The AI-Powered Retrieval-Augmented Generation (RAG) Model is a cutting-edge system designed to provide accurate, context-aware answers to user queries based on uploaded documents. It supports multiple document formats and retrieval modes, ensuring flexibility and performance. The system features a user-friendly Streamlit interface with secure authentication and session management.

---

## Project Overview

This project combines advanced AI techniques and robust software engineering to deliver a scalable and interactive document-based Q&A platform:

- **Document Processing:** Extracts and chunks text from PDFs, DOCX, and TXT files for efficient retrieval.
- **Embeddings:** Converts text chunks into vector embeddings using HuggingFace sentence-transformers.
- **Language Model:** Generates natural language answers with the Ollama LLM.
- **Retrieval Modes:**
  - *Direct Retrieval:* Fast answers based on document similarity search.
  - *Enhanced RAG:* Multi-stage refinement for improved answer quality.
  - *Hybrid Mode:* Combines document retrieval with simulated web search results.
- **Vector Stores:** Utilizes FAISS and Chroma for high-performance similarity search.
- **Web Search Simulation:** Augments answers with plausible web content when documents are insufficient.
- **User Interface:** Streamlit-based UI with customizable themes and progress indicators.
- **User Authentication:** Secure login, signup, and session management backed by MongoDB.
- **Data Storage:** MongoDB and GridFS for managing user data, documents, notebooks, and vector indexes.
- **Notebook Management:** Organize documents into notebooks with metadata, analytics, and integrated chat.
- **Performance Optimization:** Supports GPU acceleration and efficient resource management.

---

## Features

- Multiple retrieval modes for flexible question answering  
- Support for PDF, DOCX, and TXT document formats  
- Secure user authentication and session management  
- Interactive Streamlit-based user interface  
- MongoDB integration for data and vector index storage  
- Notebook and document organization with analytics  
- Voice input support via speech recognition  

---

## Skills Utilized

- Natural Language Processing (NLP) and Language Modeling  
- Document Parsing and Text Extraction  
- Vector Embeddings and Similarity Search  
- AI Model Integration and Prompt Engineering  
- Web Scraping and Content Simulation  
- Python Programming and Software Development  
- Streamlit for Web Application Development  
- Database Management with MongoDB and GridFS  
- Performance Optimization and Resource Management  
- User Authentication and Session Handling  

---

## Technologies

- Python 3.x  
- PyTorch and HuggingFace Transformers  
- Ollama Language Model API  
- LangChain Framework  
- FAISS and Chroma Vector Stores  
- PyPDF2 and python-docx  
- Streamlit  
- MongoDB and GridFS  
- BeautifulSoup and Requests  
- SpeechRecognition  

---

## Installation

1. Clone the repository:  
   ```bash
   git clone <repository-url>
   cd rag
   ```
2. (Optional) Create and activate a virtual environment:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. (Optional) Set MongoDB connection string:  
   ```bash
   export MONGODB_URI="your_mongodb_connection_string"
   ```
   Defaults to `mongodb://localhost:27017/` if not set.

2. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

3. Open the app in your browser at the URL provided by Streamlit (usually http://localhost:8501).

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
