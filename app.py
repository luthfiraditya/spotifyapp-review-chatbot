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


DATA_PATH = "data/processed/SPOTIFY_REVIEWS.csv"
FAISS_PATH = "vectorstore/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



welcome_message = """
  #### Introduction 🚀

  The chatbot is a RAG pipeline designed for discussing Spotify app user reviews from the Google Play Store.

  #### Getting started 🛠️

  1. To set up, please add your OpenAI's API key. 🔑 
  2. Type in the question you want to ask about user reviews on the Spotify app. 💬

  Please make sure to check the sidebar for more useful information. 💡

"""

usage_information = """


  1. Put your OpenAPI key on `OpenAI API Key` section. You need to give the OpenAPI key that can be generated here : https://platform.openai.com/api-keys
  
  2. If it's valid, you can choose OpenAI model that you want to Use
  
  3. Your data is not being stored anyhow by the program. Everything is recorded in a Streamlit session state and will be removed once you refresh the app. However, it must be mentioned that the **uploaded data will be processed directly by OpenAI's GPT**, which I do not have control over. 
"""



def clear_message(): 
    st.session_state.resume_list = []
    st.session_state.chat_history = [AIMessage(content=welcome_message)]




st.set_page_config(page_title="Spotify Review Chatbot")
st.title("Spotify Review Chatbot with LLM")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Welcome to the Spotify Review Chatbot!")]

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda"}
    )

if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(DATA_PATH)

if "vector_db" not in st.session_state:
    vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
    st.session_state.vector_db = vectordb


st.sidebar.title("Spotify Review Chatbot")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

with st.sidebar.expander("ℹ️ Usage Information", expanded=False):
    st.markdown(usage_information)

if not api_key:
    st.info("Please provide your OpenAI API key.")
    st.stop()

gpt_selection = st.sidebar.selectbox("GPT Model", ["gpt-4o","gpt-3.5-turbo"], placeholder="gpt-4o", key="gpt_selection")

st.sidebar.button("Clear conversation", on_click = clear_message)



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
