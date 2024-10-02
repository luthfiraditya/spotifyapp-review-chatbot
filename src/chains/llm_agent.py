import sys, httpx
sys.dont_write_bytecode = True

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai.chat_models import ChatOpenAI


class ChatBot:
    def __init__(self, api_key: str, model: str):
        self.llm = ChatOpenAI(api_key=api_key, model=model)

    def classify_query(self, question: str):
        """
        Use the LLM to classify the type of question being asked (e.g., sentiment, trends, or general).
        """
        system_message = SystemMessage(
        content="""You are an expert in analyzing user reviews. Based on the following question, classify the type of analysis required.
        The possible types are: 'sentiment', 'trends', 'comparison', or 'general'. Based on the question, return only the type of analysis required.
        Use the following examples as a guide:
        
        Examples:
        
        1. Question: "What do users think about the new music recommendation feature?"
           Classification: 'sentiment'

        2. Question: "What are the emerging trends in the latest user reviews?"
           Classification: 'trends'

        3. Question: "How does our app compare to Apple Music in terms of features?"
           Classification: 'comparison'

        4. Question: "What do users typically look for in a music streaming app?"
           Classification: 'general'

        5. Question: "What are the specific features or aspects that users appreciate the most in our application?"
           Classification: 'sentiment'
        """
    )

        user_message = HumanMessage(content=f"Classify this question: {question}")

        response = self.llm.invoke([system_message, user_message])
        query_type = response.content.strip().lower()
        
        return query_type

    def handle_query(self, question: str, docs: list, query_type: str):
        """
        Based on the classified query type (sentiment, trends, etc.), handle the analysis and prompt generation dynamically.
        """
        context = "\n\n".join(doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs)

        if query_type == "sentiment":
            system_message = SystemMessage(
                content="""
                You are an expert in analyzing Spotify reviews to determine user sentiment (positive or negative).
                Based on the following reviews, identify the general sentiment and provide your analysis.
                """
            )

        elif query_type == "trends":
            system_message = SystemMessage(
                content="""
                You are an expert in analyzing Spotify reviews to detect emerging trends.
                Based on the following reviews, identify key trends that have emerged recently.
                """
            )
    
        
        else:
            system_message = SystemMessage(
                content="""
                You are an expert in analyzing Spotify reviews.
                Based on the following reviews, provide an insightful response to the user's question.
                """
            )

        user_message = HumanMessage(content=f"Context: {context}, Question: {question}")
        
        stream = self.llm.stream([system_message, user_message])

        return stream
