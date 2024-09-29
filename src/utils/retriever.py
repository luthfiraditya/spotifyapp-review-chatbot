import sys
sys.dont_write_bytecode = True

from typing import List
from pydantic import BaseModel, Field

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema.agent import AgentFinish
from langchain.tools.render import format_tool_to_openai_function
import pandas as pd
from datetime import datetime


DATA_PATH = "data/processed/applicant_resume.csv"  
FAISS_PATH = "vectorstore/"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RAG_K_THRESHOLD = 5


class RagRetriever():
    def __init__(self, vectorstore_db, df):
        self.vectorstore = vectorstore_db
        self.df = df

    def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50):
        """
        Combines multiple ranked lists of documents into a single ranked list using the Reciprocal Rank Fusion (RRF) method.
        """
        fused_score = {}
        for doc_list in document_rank_list:
            for rank, (doc, _) in enumerate(doc_list.items()):
                if doc not in fused_score:
                    fused_score[doc] = 0
                else:
                    fused_score[doc] += 1 / (rank+k)
        reranked_results = {doc: score for doc, score in sorted(fused_score.items(), key=lambda x: x[1], reverse=True)}
        return reranked_results

    def __retrieve_docs_id__(self, question: str, k=50):
        """
        Retrieves document IDs and their similarity scores based on a query.
        """
        docs_score = self.vectorstore.similarity_search_with_score(question, k)
        docs_score = {str(doc.metadata["review_id"]): score for doc, score in docs_score}
        return docs_score

    def retrieve_id_and_rerank(self, subquestion_list: list):
        """
        Retrieve and rerank document IDs based on multiple subqueries.
        """
        document_rank_list = []
        for subquestion in subquestion_list:
            document_rank_list.append(self.__retrieve_docs_id__(subquestion, RAG_K_THRESHOLD))

        reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
        return reranked_documents

    def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5):
        """
        Fetch the actual content of documents (resumes or reviews) based on their IDs and corresponding scores.
        """
        id_resume_dict = dict(zip(self.df["review_id"].astype(str), self.df["review_text"]))
        retrieved_ids = list(sorted(doc_id_with_score, key=doc_id_with_score.get))[:threshold]
        retrieved_documents = [id_resume_dict[id] for id in retrieved_ids]
        for i in range(len(retrieved_documents)):
            retrieved_documents[i] = "Review ID " + retrieved_ids[i] + "\n" + retrieved_documents[i]
        return retrieved_documents

    def retrieve_by_time_and_rating(self, subquestion_list: list, rating_threshold: float = None, time_window_days: int = None):
        """
        Retrieves documents by filtering based on satisfaction (review_rating) and time (review_timestamp).
        """
        document_rank_list = []
        current_time = datetime.now()

        for subquestion in subquestion_list:
            docs_score = self.__retrieve_docs_id__(subquestion, RAG_K_THRESHOLD)
            filtered_docs = {}

            for doc_id, score in docs_score.items():
                review = self.df[self.df["review_id"] == doc_id].iloc[0]
                review_rating = float(review["review_rating"])
                review_timestamp = datetime.strptime(review["review_timestamp"], "%Y-%m-%d %H:%M:%S")


                # Filter based on rating and time
                if (rating_threshold is not None and review_rating <= rating_threshold) or \
                   (time_window_days is not None and (current_time - review_timestamp).days <= time_window_days):
                    filtered_docs[doc_id] = score

            document_rank_list.append(filtered_docs)

        reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
        return reranked_documents


class SelfQueryRetriever(RagRetriever):
    def __init__(self, vectorstore_db, df):
        super().__init__(vectorstore_db, df)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in analyzing Spotify reviews."),
            ("user", "{input}")
        ])

        self.meta_data = {
            "rag_mode": "",
            "query_type": "no_retrieve",
            "extracted_input": "",
            "subquestion_list": [],
            "retrieved_docs_with_scores": []
        }

    def retrieve_docs(self, question, llm, rag_mode: str):
        """
        Extended functionality to handle specific questions like those related to time (emerging trends) or dissatisfaction (rating).
        """
        if "emerging trends" in question.lower():  # Handle time-based trends
            print("Handling time-based question...")
            return self.retrieve_by_time_and_rating([question], time_window_days=30)

        elif "dissatisfaction" in question.lower():  
            print("Handling dissatisfaction-based question...")
            return self.retrieve_by_time_and_rating([question], rating_threshold=2.0)

        else:
            subquestion_list = [question]
            if rag_mode == "RAG Fusion":
                subquestion_list += llm.generate_subquestions(question)

            self.meta_data["subquestion_list"] = subquestion_list
            retrieved_ids = self.retrieve_id_and_rerank(subquestion_list)
            self.meta_data["retrieved_docs_with_scores"] = retrieved_ids
            retrieved_resumes = self.retrieve_documents_with_id(retrieved_ids)
            return retrieved_resumes
