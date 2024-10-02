import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from src.utils.retriever import SelfQueryRetriever  

# Define constants
DATA_PATH = "data/raw_data/SPOTIFY_REVIEWS.csv"  
FAISS_PATH = "vectorstore/"  
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})

vectorstore_db = FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

df = pd.read_csv(DATA_PATH)


api_key = "YOUR-API-KEY"
model_name = "gpt-4o" 
retriever = SelfQueryRetriever(vectorstore_db, df, api_key, model_name)

def test_trend_query():
    print("Test Case: Trend Query")
    question = "What are the emerging trends in reviews in the last 90 days?"
    docs = retriever.retrieve_by_time_and_rating(question, time_window_days=90)
    print("Retrieved Documents:")
    for doc, score in docs:
        print(f"Review ID: {doc.metadata['review_id']}, Rating: {doc.metadata['review_rating']}, Timestamp: {doc.metadata['review_timestamp']}")
        print(f"Content: {doc.page_content}")
        print("-" * 80)

def test_rating_query():
    print("Test Case: Rating Query")
    question = "Find me reviews with a rating higher than 3"
    docs = retriever.retrieve_by_time_and_rating(question, rating_threshold=3)
    print("Retrieved Documents:")
    for doc, score in docs:
        print(f"Review ID: {doc.metadata['review_id']}, Rating: {doc.metadata['review_rating']}, Timestamp: {doc.metadata['review_timestamp']}")
        print(f"Content: {doc.page_content}")
        print("-" * 80)

if __name__ == "__main__":
    test_trend_query()
    print("\n")
    test_rating_query()
