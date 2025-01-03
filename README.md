# Query US Code

This Streamlit application allows users to query the entire US Code.

## Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory with the following variables:
```
VOYAGE_API_KEY=your_voyage_api_key
GOOGLE_API_KEY=google_api_key
```

### 4. Run the Application
```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501`

## Features
- Multiple retriever options:
  - Normal Vector Retrieval
  - BM25 Retrieval
  - Ensemble Retrieval
  - Compression Retrieval with Reranking
- Natural language querying of the use code

## Note
This application requires valid API keys for VoyageAI and Google's Gemini. Make sure these are properly configured in your `.env` file before running the application.