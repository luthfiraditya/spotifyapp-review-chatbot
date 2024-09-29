import sys
sys.dont_write_bytecode = True

from utils.retriever import SelfQueryRetriever  # Import the modified retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd

# Constants
FAISS_PATH = "vectorstore/"  # Path to saved FAISS vector store
DATA_PATH = "data/raw_data/SPOTIFY_REVIEWS.csv"  # Path to the reviews CSV file
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load your data
def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

# Load FAISS vector store properly (with metadata)
def load_faiss_index(faiss_path):
    # Load the FAISS vector store (not just the raw FAISS index)
    vectorstore_db = FAISS.load_local(faiss_path, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),allow_dangerous_deserialization=True )
    return vectorstore_db

# Main function to test the retriever
def test_retriever():
    # Load the FAISS vector store
    index = load_faiss_index(FAISS_PATH)
    
    # Load the DataFrame with reviews
    df = load_data(DATA_PATH)
    
    # Initialize the retriever with the FAISS vector store and data
    retriever = SelfQueryRetriever(vectorstore_db=index, df=df)
    
    # Test query 1: Emerging trends based on recent reviews (TIME)
    question_1 = "Can you identify emerging trends or patterns in recent user reviews that may impact our product strategy?"
    result_1 = retriever.retrieve_docs(question_1, llm=None, rag_mode="RAG Fusion")
    print("\n--- Test 1: Emerging Trends ---")
    for doc in result_1:
        print(doc)
    
    # Test query 2: Reasons for dissatisfaction (low ratings)
    question_2 = "What are the primary reasons users express dissatisfaction with Spotify?"
    result_2 = retriever.retrieve_docs(question_2, llm=None, rag_mode="RAG Fusion")
    print("\n--- Test 2: Dissatisfaction ---")
    for doc in result_2:
        print(doc)

if __name__ == "__main__":
    test_retriever()
