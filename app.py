import os
import openai
import time
import pandas as pd
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores.faiss import DistanceStrategy
from src.utils.retriever import SelfQueryRetriever
from src.chains.llm_agent import ChatBot


DATA_PATH = "data/raw_data/SPOTIFY_REVIEWS.csv"
FAISS_PATH = "vectorstore/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="Spotify Review Chatbot")
st.title("Spotify Review Chatbot with LLM")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Welcome to the Spotify Review Chatbot!")]

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )

if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(DATA_PATH)

if "vector_db" not in st.session_state:
    vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
    st.session_state.vector_db = vectordb


st.sidebar.title("Spotify Review Chatbot")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.info("Please provide your OpenAI API key.")
    st.stop()

gpt_selection = st.sidebar.selectbox("GPT Model", ["gpt-4o","gpt-3.5-turbo"], placeholder="gpt-4o", key="gpt_selection")

if "retriever" not in st.session_state:
    st.session_state.retriever = SelfQueryRetriever(st.session_state.vector_db, st.session_state.df, api_key, gpt_selection)

user_query = st.chat_input("Ask me anything about Spotify user reviews!")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

llm = ChatBot(api_key=api_key, model=gpt_selection)

if user_query is not None and user_query != "":
    with st.chat_message("Human"):
        st.write(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("AI"):
        with st.spinner("Generating a response..."):
            retriever = st.session_state.retriever
            docs = retriever.retrieve_docs(question=user_query)
            prompt_cls = retriever.meta_data["query_type"]

            stream_message = llm.handle_query(
                question=user_query,
                docs=docs,
                query_type=retriever.meta_data["query_type"]
            )


            ai_message_placeholder = st.empty()
            full_response = ""


            with ai_message_placeholder:
                for chunk in stream_message:
                    full_response += chunk.content
                    ai_message_placeholder.write(full_response)

            end = time.time()

            st.session_state.chat_history.append(AIMessage(content=full_response))
else:
    st.info("Please ask a question to continue.")
