from src.utils.retriever import SelfQueryRetriever  # Import your defined classes
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pandas as pd

DATA_PATH = "data/raw_data/SPOTIFY_REVIEWS.csv"
FAISS_PATH = "vectorstore/"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



def print_stream_results(stream):
    """
    Helper function to print the results from a streaming generator.
    """
    for chunk in stream:
        print(chunk.content, end="")  
    print("\n")          


def load_faiss_vectorstore():
    """
    Loads the FAISS vector store from the saved index.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"})
    
    vectorstore_db = FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return vectorstore_db
def test_llm_retriever():
    df = pd.read_csv(DATA_PATH)
    vectorstore_db = load_faiss_vectorstore()
    api_key = "YOUR-API-KEY"
    model_name = "gpt-4o" 

    retriever = SelfQueryRetriever(vectorstore_db=vectorstore_db, df=df, api_key=api_key, model=model_name)
    question_1 = "What are the specific features or aspects that users appreciate the most in our application?"
    question_2 = "Can you identify emerging trends or patterns in recent user reviews that may impact our product strategy?"
    question_3 = "What are the primary reasons users express dissatisfaction with Spotify?"
    '''
    # Retrieve responses for each question
    print("Test Case 1: Sentiment-based question")
    result_1 = retriever.retrieve_docs(question_1)
    print(f"LLM Response for Sentiment: {result_1}\n")
    print_stream_results(result_1)
    '''
    print("Test Case 2: Time-based trends question")
    result_2 = retriever.retrieve_docs(question_2)
    print(f"LLM Response for Trends: {result_2}\n")
    print_stream_results(result_2)


    '''
    print("Test Case 3: Dissatisfaction-based question")
    result_3 = retriever.retrieve_docs(question_3)
    print(f"LLM Response for Dissatisfaction: {result_3}\n")
    print_stream_results(result_3)
    '''
if __name__ == "__main__":
    test_llm_retriever()
