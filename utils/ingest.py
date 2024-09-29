import pandas as pd
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "data/raw_data/SPOTIFY_REVIEWS.csv"
FAISS_PATH = "vectorstore/"


def combine_metadata_with_text(df):
    df['combined_text'] = df.apply(lambda row: f"Rating: {row['review_rating']}, Date: {row['review_timestamp']}, Review: {row['review_text']}", axis=1)
    return df


def ingest_to_vector(df, content_column, embedding_model):
    """
    Ingests data from a DataFrame into a vector database.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to ingest.
        content_column (str): The name of the column containing the text content.
        embedding_model (HuggingFaceEmbeddings): The embedding model to use.
        
    Returns:
        FAISS: The created FAISS vector database.
    """

    df[content_column].fillna("", inplace=True)
    loader = DataFrameLoader(df, page_content_column=content_column)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=500)
    document_chunks = text_splitter.split_documents(documents)

    vectorstore_db = FAISS.from_documents(
        document_chunks,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    
    return vectorstore_db

def load_and_ingest_csv(csv_path, content_column, embedding_model):
    """
    Loads data from a CSV and ingests it into a vector database.
    
    Args:
        csv_path (str): The path to the CSV file.
        content_column (str): The column in the CSV file containing the text content.
        embedding_model (HuggingFaceEmbeddings): The embedding model to use.
        
    Returns:
        FAISS: The FAISS vector store created from the CSV data.
    """
    try:
        df = pd.read_csv(csv_path)
        
        df['review_text'] = df['review_text'].str.lower()
        df['review_text'] = df['review_text'].str.replace('[^\w\s]', '', regex=True)
        df = df[['review_id', 'review_text', 'review_rating', 'review_timestamp']]
        df = df[:1000]

        df = combine_metadata_with_text(df)

        vectordb = ingest_to_vector(df, content_column=content_column, embedding_model=embedding_model)
        
        vectordb.save_local(FAISS_PATH)
        
        print(f"Vector database saved at '{FAISS_PATH}'")
        return vectordb
    
    except FileNotFoundError as e:
        print(f"Error on File: {e}")
    except Exception as e:
        print(f"error occurred : {e}")

'''
def main():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})
    load_and_ingest_csv(DATA_PATH, content_column='review_text', embedding_model=embedding_model)

if __name__ == "__main__":
    main()

'''
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})
vectorstore_db = load_and_ingest_csv(DATA_PATH, content_column='review_text', embedding_model=embedding_model)

query = "What do users think about the music recommendation feature?"

results = vectorstore_db.similarity_search(query)

for result in results:
    print(f"Review ID: {result.metadata['review_id']}")
    print(f"Review Text: {result.page_content}")
    print("\n---\n")

