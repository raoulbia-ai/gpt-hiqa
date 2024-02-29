from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(), override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


# Initialize shared instances
# llm_instance = OpenAI(temperature=0, model="gpt-3.5-turbo")
# embedding_instance = OpenAIEmbedding(model="text-embedding-ada-002")

llm_instance = OpenAI(temperature=0, model="gpt-4")
# embedding_instance = OpenAIEmbedding(model="text-embedding-3-small")
embedding_instance = OpenAIEmbedding(model="text-embedding-ada-002")

