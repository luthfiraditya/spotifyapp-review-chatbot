import sys
sys.dont_write_bytecode = True

from typing import List
from pydantic import BaseModel, Field
import pandas as pd
from datetime import datetime, timedelta

from src.chains.llm_agent import ChatBot

DATA_PATH = "data/processed/SPOTIFY_REVIEWS.csv"  
FAISS_PATH = "vectorstore/"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RAG_K_THRESHOLD = 5


class RagRetriever():
    def __init__(self, vectorstore_db, df):
        self.vectorstore = vectorstore_db
        self.df = df

    def retrieve_documents_with_id(self, docs_score: list, threshold=5):
        """
        Fetch the actual content of documents (reviews) based on their scores.
        Now handling `docs_score` as a list of tuples (doc, score).
        """
        id_resume_dict = dict(zip(self.df["review_id"].astype(str), self.df["review_text"]))
        retrieved_ids = [str(doc.metadata["review_id"]) for doc, score in docs_score[:threshold]]
        retrieved_documents = [id_resume_dict[doc_id] for doc_id in retrieved_ids]
        
        for i in range(len(retrieved_documents)):
            retrieved_documents[i] = f"Review ID: {retrieved_ids[i]}\nReview: {retrieved_documents[i]}"
        
        return retrieved_documents

    def retrieve_by_time_and_rating(self, question: str, rating_threshold: float = None, time_window_days: int = None):
        """
        Retrieves documents by filtering based on satisfaction (review_rating) and time (review_timestamp).
        Filters by the last `time_window_days` and/or ratings below `rating_threshold`, based on max review_timestamp in the dataset.
        """
        docs_score = self.vectorstore.similarity_search_with_score(question, RAG_K_THRESHOLD)
        filtered_docs = []
        
        # Find the maximum timestamp from the dataset (to simulate the current date)
        max_timestamp = pd.to_datetime(self.df['review_timestamp']).max()

        # Calculate time threshold relative to the max_timestamp
        time_threshold = max_timestamp - timedelta(days=time_window_days) if time_window_days else None

        for doc, score in docs_score:
            review_id = doc.metadata['review_id']
            review = self.df[self.df["review_id"] == review_id].iloc[0]
            review_rating = float(review["review_rating"])
            review_timestamp = pd.to_datetime(review["review_timestamp"])

            # Filter based on rating and timestamp relative to max_timestamp
            if (rating_threshold is not None and review_rating <= rating_threshold) or \
            (time_window_days is not None and review_timestamp >= time_threshold):
                filtered_docs.append((doc, score))

        return filtered_docs


class SelfQueryRetriever(RagRetriever):
    def __init__(self, vectorstore_db, df, api_key, model):
        super().__init__(vectorstore_db, df)
        self.chatbot = ChatBot(api_key, model)
        self.meta_data = {}

    def retrieve_docs(self, question: str, history=None):
        """
        Dynamically retrieve documents based on query classification (sentiment, trends, or general).
        """
        query_type = self.chatbot.classify_query(question)
        self.meta_data["query_type"] = query_type
        print(f"Classified query as: {query_type}")
        
        # If the query type is related to trends, retrieve data from the last 3 months based on max date in the data
        if query_type == "trends":
            three_months_in_days = 500  # Last 3 months in days
            print("Retrieving using time and rating filter for trends...")
            docs = self.retrieve_by_time_and_rating(question, time_window_days=three_months_in_days)
        else:
            docs = self.retrieve_documents_with_id(self.vectorstore.similarity_search_with_score(question, RAG_K_THRESHOLD))

        response = self.chatbot.handle_query(question, docs, query_type)
        
        return response

