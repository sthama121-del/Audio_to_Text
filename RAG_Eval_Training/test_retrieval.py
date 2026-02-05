# Save this as test_retrieval.py
import config
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Create embedding model
embedding_model = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    openai_api_key=config.OPENAI_API_KEY,
)

# Load existing vector store
vector_store = Chroma(
    collection_name=config.CHROMA_COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)

# Test count
print(f"Total vectors in store: {vector_store._collection.count()}")

# Test retrieval
query = "remote work policy"
results = vector_store.similarity_search(query, k=3)
print(f"Query: {query}")
print(f"Results: {len(results)}")