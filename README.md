# RAG Legal Assistant

A **Retrieval-Augmented Generation (RAG)**-based pipeline designed to streamline legal workflows by automating tasks such as legal document summarization, case discovery, legal drafting, and query resolution. This project leverages state-of-the-art Natural Language Processing (NLP) techniques and a multi-agent architecture to provide an intuitive solution for legal professionals.

## Features

- **Legal Document Summarization**: Generate concise summaries of lengthy legal documents.
- **Case Discovery**: Retrieve and analyze relevant legal cases based on user queries.
- **Legal Drafting**: Automate the creation of professional and structured legal documents.
- **Query Answering**: Provide accurate responses to legal questions, leveraging the context of uploaded documents or query inputs.
- **Multi-Agent Architecture**: Ensure modular and scalable solutions to meet diverse legal needs.

## System Overview

### 1. Core Components
- **User Input**: Accepts text-based queries or uploads of legal documents (e.g., case files, contracts).
- **Query Decomposition**: Breaks down complex user inputs into manageable subqueries.
- **Fusion RAG Framework**: Combines advanced retrieval and generation techniques to produce accurate and contextually relevant outputs.

### 2. Technologies Used
- **Retriever**: Utilizes FAISS for indexing and `all-mpnet-base-v2` for semantic embeddings.
- **Generator**: Employs `microsoft/Phi-3-mini-4k-instruct` for generating summaries and responses.
- **Intent Classification**: Leverages `nlpaueb/legal-bert-base-uncased` fine-tuned on legal datasets.

### 3. Dataset
- Supreme Court of India case files spanning the past two years.
- Extracted attributes include case titles, dates, involved parties, and legal categories.
- Preprocessed for metadata extraction and tokenized for efficient retrieval.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Irwindeep/rag-legal-assistant.git
   cd rag-legal-assistant
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and configure the necessary models:
   - Use Hugging Face Hub to download `nlpaueb/legal-bert-base-uncased`, `all-mpnet-base-v2`, and `microsoft/Phi-3-mini-4k-instruct`.
   - Configure these models in the project settings.

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

- Open the Streamlit application in your browser.
- Enter your query or upload a legal document.
- Select the task you want to perform (e.g., summarization, case discovery, legal drafting).
- View and download the generated results.
