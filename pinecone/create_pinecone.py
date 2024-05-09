import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

pc = pinecone.Pinecone(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)

embeddings = OpenAIEmbeddings()

vector_store = pc.VectorStore(
    os.getenv('PINECONE_INDEX_NAME'),
)

