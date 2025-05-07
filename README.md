# AI-Powered-RAG-Model

## About
AI-Powered RAG Model with Secure Document-Based Q&amp;A. This project implements a Retrieval-Augmented Generation (RAG) model that processes PDFs and answers user queries efficiently. The system supports three retrieval modes: Direct Retrieval, Enhanced RAG (online enrichment), and Hybrid Mode. It also features a user interface with authentication.

## Features
- Three Retrieval Modes:
  - Direct Retrieval
  - Enhanced RAG (online enrichment)
  - Hybrid Mode
- User Interface built with Streamlit
- User Authentication and Session Management
- Secure document-based question answering
- MongoDB integration for data storage

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. (Optional) Set the MongoDB connection string environment variable:
   ```bash
   export MONGODB_URI="your_mongodb_connection_string"
   ```
   If not set, the app defaults to `mongodb://localhost:27017/`.

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Access the app in your browser at the URL provided by Streamlit (usually http://localhost:8501).

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
