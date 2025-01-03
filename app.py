import streamlit as st
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import pickle

load_dotenv()

def load_model():
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED:
        HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
        HarmBlockThreshold.BLOCK_ONLY_HIGH
    }

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",safety_settings=safety_settings)
    return llm


def initialize_retrievers():
    load_dotenv()
    
    # Initialize embeddings and vector store
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=voyage_api_key, 
        model="voyage-3-lite"
    )
    
    vector_store = LanceDB(embedding=embeddings,table_name="usa_code",uri = "databases/lance",distance="cosine")

    # Normal retriever
    normal_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    bm_25_db_path = "databases/bm25_retriever.pkl"

    with open(bm_25_db_path, 'rb') as f:
        bm25_retriever = pickle.load(f)

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever.with_config({"kwargs": {"k": 1}}),
            normal_retriever.with_config({"kwargs": {"k": 1}})
        ],
        weights=[0.5, 0.5]
    )
    
    # Compression retriever
    compressor = VoyageAIRerank(
        model="rerank-2-lite", 
        voyageai_api_key=voyage_api_key, 
        top_k=2
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=normal_retriever
    )
    
    return {
        "Normal": normal_retriever,
        "BM25": bm25_retriever,
        "Ensemble": ensemble_retriever,
        "Cross-Encoder": compression_retriever
    }

import re

def parse_markdown_tags(text: str) -> str:
    pattern = r'```markdown\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return ""

def format_docs(docs):
    """Extract xml_content from document metadata and join them."""
    # Extract xml_content from each document's metadata
    xml_contents = [doc.metadata.get("xml_content", "") for doc in docs]
    # Join all xml contents with newlines
    return "\n\n".join([content for content in xml_contents if content])

def main():
    st.title("Query USA Code")
    
    # Initialize retrievers and store in session state if not already done
    if 'retrievers' not in st.session_state:
        st.session_state.retrievers = initialize_retrievers()
    
    # Retriever selection
    retriever_type = st.selectbox(
        "Select Retriever Type",
        options=list(st.session_state.retrievers.keys())
    )
    
    # Query input
    query = st.text_input("Enter your query about legislation:")
    
    if st.button("Search"):
        if query:
            with st.spinner('Processing query...'):
                # Initialize model and prompt
                model = load_model()
                
                template = """You are analyzing the United States Code and related legislative documents. Below is relevant content from the US Code in XML format:
                            {context}

                            Based solely on the provided content, compose a structured response surrounded by ```markdown``` tags that:

                            1. Begins with the relevant US Code title and section reference
                            2. Includes source credit in proper format:
                            - For main law: (Pub. L. XX-XXX, §X, Date, XX Stat. XXXX)
                            - For amendments: As amended Pub. L. XX-XXX, §X, Date, XX Stat. XXXX
                            3. Explains provisions clearly with proper markdown formatting
                            4. Uses the following structure:

                            ```markdown
                            # [Title Number] U.S.C. § [Section Number] - [Section Title]

                            ## Source
                            [Source credits in standard legal citation format]

                            ## Current Law
                            [Main content explanation]

                            ## Legislative History
                            [Relevant amendments and changes]

                            ## Important Notes
                            [Relevant findings and additional provisions]
                            ```

                            Ensure all responses maintain consistent markdown formatting and are wrapped in markdown tags.
                            Question: {question}
                            Answer: """

                prompt = ChatPromptTemplate.from_template(template)     
                
                # Create chain with selected retriever
                chain = (
                    {"context": st.session_state.retrievers[retriever_type] | format_docs, 
                     "question": RunnablePassthrough()}
                    | prompt
                    | model
                    | StrOutputParser()
                    | parse_markdown_tags
                )
                
                # Get response
                response = chain.invoke(query)
                
                # Display response
                with st.container():
                    st.markdown(response, unsafe_allow_html=True)

if __name__ == "__main__":
    main()