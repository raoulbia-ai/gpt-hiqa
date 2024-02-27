from src.document_processor import DocumentProcessor
from src.query_manager import QueryManager
from config import llm_instance, embedding_instance, HF_TOKEN
from src.pincone_manager import PineconeManager
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
import numpy as np
from config import OPENAI_API_KEY
import torch

from llama_index.core import Settings

# embed_model = OpenAIEmbedding(embed_batch_size=10)
# Settings.embed_model = embed_model
# embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
client = OpenAI()
MODEL = "text-embedding-ada-002"


# we need an embedding of dim 768 hence we use all-mpnet-base-v2
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", use_auth_token=HF_TOKEN)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", use_auth_token=HF_TOKEN)


# def insert_embeddings(index_name, embeddings):
#    vectorstore = pc_manager.query(index_name, embeddings)
#    vectorstore = pc_manager.upsert(chunks, embeddings, index_name=index_name)
#    return vectorstore

def encode_question(question_text):
    # Directly encode the question text to embeddings
    # embeddings = model.encode(question_text) # This already returns a numpy array
    # embeddings = embeddings.tolist()
    # embeddings = embed_model.get_text_embedding(question_text)
    embeddings = client.embeddings.create(
        input=[
            "Sample document text goes here",
            "there will be several phrases in each batch"
        ], model=MODEL
    )
    embeddings = [record.embedding for record in embeddings.data]
    return embeddings


def get_or_create_answer(query_manager, question_text):
    encoded_question = encode_question(question_text)
    print(encoded_question)

    vector_index = PineconeManager(index_name="my-test-index")
    # similar_questions = vector_index.query(
    #     vector=question_text,
    #     top_k=3)
    # print(similar_questions)

    xq = client.embeddings.create(input=question_text, model=MODEL).data[0].embedding
    res = vector_index.query([xq[0]], top_k=5)
    for match in res['matches']:
        print(f"{match['score']:.2f}: {match['metadata']['text']}")

    # if similar_questions:
    #     print("Generated answer from Pinecone")
    #     return pc_manager.query(similar_questions[0])
    # else:
    #     print("Generated answer from OpenAI")
    #     # answer = generate_answer_with_openai(question_text)
    #     # pc_manager.upsert(question_text, answer, encoded_question)
    #     # return answer
    #     master_agent = query_manager.get_answer()
    #     return master_agent.query(question_text)

# def generate_answer_with_openai(question_text):
#     # Your code to query OpenAI API and return the answer
#     return "Generated answer from OpenAI"


def main():


    # Initialize the DocumentProcessor and process documents
    document_processor = DocumentProcessor(llm_instance, embedding_instance)
    document_processor.load_documents()  # Ensure this method is implemented to load your documents
    document_processor.build_agents()

    # Initialize QueryManager with the DocumentProcessor instance
    query_manager = QueryManager(llm_instance, embedding_instance, document_processor)
    query_manager.build_tools()


    # Submit a text query
    query_text = "list all centers"
    response = get_or_create_answer(query_manager, query_text)



if __name__ == "__main__":
    main()
