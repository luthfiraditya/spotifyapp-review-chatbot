import chromadb
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI

chroma_client = chromadb.PersistentClient(path="chromadb_store")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(collection_name="spotify_reviews", client=chroma_client, embedding_function=embedding_model)

metadata_field_info = [
    AttributeInfo(
        name="review_rating",
        description="The rating of the review",
        type="float",
    ),
    AttributeInfo(
        name="review_timestamp",
        description="The timestamp of the review",
        type="string",
    ),
    AttributeInfo(
        name="review_id",
        description="The unique ID of the review",
        type="string",
    ),
]

document_content_description = "Text of the review for a music streaming platform like Spotify"

llm = OpenAI(temperature=0, api_key = "YOUR-API-KEY")  

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
)

query = "Find me reviews where the rating of review is lower than 3 and the year is of timestamp is 2023"
results = retriever.invoke(query)

for result in results:
    print(f"Review ID: {result.metadata['review_id']}")
    print(f"Review Rating: {result.metadata['review_rating']}")
    print(f"Review Timestamp: {result.metadata['review_timestamp']}")
    print(f"Review Text: {result.page_content}")
    print("\n---\n")
