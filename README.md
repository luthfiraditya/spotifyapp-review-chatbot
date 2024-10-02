
# Spotify Review Chatbot

This project implements a chatbot that processes and analyzes user reviews of Spotify using a **Retrieval-Augmented Generation (RAG)** pipeline. The chatbot allows you to retrieve insights from review data and analyze trends, sentiments, and specific user queries using **LangChain**, **FAISS**, and **OpenAI** models.

## Table of Contents

- [Spotify Review Chatbot](#spotify-review-chatbot)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Technologies Used](#technologies-used)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Clone the Repository](#clone-the-repository)
    - [Create and Activate Virtual Environment (venv)](#create-and-activate-virtual-environment-venv)
    - [Install Dependencies](#install-dependencies)
  - [Usage](#usage)
    - [Preprocess the data using Makefile](#preprocess-the-data-using-makefile)
    - [Running the Application (virtual environment)](#running-the-application-virtual-environment)
    - [Running the Application (Docker)](#running-the-application-docker)
    - [Querying the Chatbot](#querying-the-chatbot)
  - [Testing](#testing)
  - [NOTES](#notes)

## Project Overview

The Spotify Review Chatbot is designed to help users explore and analyze trends and insights from Spotify reviews using natural language queries. The chatbot employs a **RAG** approach, combining **FAISS vector search** and **LLM** (OpenAI Model) capabilities to answer queries about user reviews.

For instance, you can query:
- Emerging trends in the reviews for the last 90 days.
- Sentiments about a specific feature, such as the "music recommendation" feature.
- Reviews filtered by ratings or timestamps.

## Features

- **Review Analysis**: Perform natural language queries on user reviews and retrieve relevant responses based on the data.
- **Sentiment Analysis**: Classify user queries as sentiment-related and provide sentiment analysis of the reviews.
- **Trend Detection**: Detect emerging trends in reviews by analyzing the most recent reviews.
- **Rating-based Filtering**: Retrieve reviews with specific rating filters (e.g., reviews with ratings greater than 3).

## Technologies Used

- **LangChain** (framework): Used for building the RAG pipeline.
- **FAISS** (vector db): An open-source vector similarity search engine used for efficient review retrieval.
- **OpenAI GPT** (LLM model): For generating answers based on the retrieved review content.
- **HuggingFace Embeddings** (embedding model): For embedding reviews into vector space.
- **Pandas** : For data processing and management.
- **Streamlit** (web deployment framework): For building a user-friendly UI to interact with the chatbot.
- **Docker** (containerization): For build, deploy, run, update and manage containerized applications

## Installation

### Prerequisites

Before you start, make sure you have the following installed:

- Python 3.10 or higher
- OpenAI API key (sign up at https://platform.openai.com/account/api-keys)

### Clone the Repository

```bash
git clone https://github.com/luthfiraditya/spotifyapp-review-chatbot.git
cd spotify-review-chatbot
```

### Create and Activate Virtual Environment (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate   # For Windows: .venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Preprocess the data using Makefile
1. Place the initial dataset in data/raw_data.
2. Run `make preprocess`. This will result in preprocessed data located in data/preprocess.
3. Run `make ingest`. This will perform embedding and store the data in the vector database (Faiss).

### Running the Application (virtual environment)

To start the chatbot, run the following command:

```bash
streamlit run app.py
```

This will launch a web-based interface where you can interact with the chatbot.


### Running the Application (Docker)

1. `docker-compose up -d`


### Querying the Chatbot

You can ask natural language questions like:

- "What do users say about our app's ability to help them discover new music?"
- "How do users feel about the app’s personalization and recommendation algorithms?"
- "Which features of our app do users find most difficult to use?"
- "Are there any specific improvements users would like to see in the app's mobile experience?"

The chatbot will process the query, retrieve relevant documents, and respond based on the reviews stored in the vector database.



## Testing

You can test the retriever functionality by running the provided `test_retriever.py` script:

```bash
python test_retriever.py
```

This script includes test cases for trend queries, rating queries, and sentiment queries, and will help you validate the performance of the RAG pipeline.


## NOTES
- The preprocessing process is not yet optimal, especially for review_text, as there are some review texts that are not in English.
- The chatbot results can be found in the folder result/llm_response/.
- I haven't had the chance to add the evaluation process yet. Initially, I planned to use RAGAS, but due to time constraints, I wasn't able to implement it.