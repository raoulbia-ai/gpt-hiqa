from src.document_processor import DocumentProcessor
from src.query_manager import QueryManager
from config import llm_instance, embedding_instance, HF_TOKEN
# from src.pincone_manager import PineconeManager
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
import time
import numpy as np
from config import OPENAI_API_KEY, PINECONE_API_KEY
import torch



from llama_index.core import Settings

# embed_model = OpenAIEmbedding(embed_batch_size=10)
# Settings.embed_model = embed_model
# embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
client = OpenAI()
MODEL = "text-embedding-ada-002"


# IF we need an embedding of dim 768 hence then we can use all-mpnet-base-v2
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", token=HF_TOKEN)
# model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", token=HF_TOKEN)

# pc = PineconeManager(index_name="my-test-index")

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
        input=[question_text], model=MODEL
    )
    query_embedding = [record.embedding for record in embeddings.data]
    return query_embedding


def get_or_create_answer(query_manager, question_text):
    encoded_question = encode_question(question_text)
    # print(encoded_question)

    pc_index_query_response = pc.index.query(
        vector=encoded_question,
        top_k=3,
        include_values=False)
    sim_scores = []
    for sq in pc_index_query_response['matches']:
        print(sq['id'], sq['score'])
        sim_scores.append(sq['score'])

    # xq = client.embeddings.create(input=question_text, model=MODEL).data[0].embedding
    # res = vector_index.query([xq[0]], top_k=5)
    # for match in res['matches']:
    #     print(f"{match['score']:.2f}: {match['metadata']['text']}")

    if max(sim_scores) > 90:
        print("Generated answer from Pinecone")
        print(pc_index_query_response['matches'])
        # return pc.index.query(similar_questions['matches'])

    else:
        print("Generated answer from OpenAI")
        # answer = generate_answer_with_openai(question_text)
        # pc_manager.upsert(question_text, answer, encoded_question)
        # return answer
        master_agent = query_manager.get_answer()
        answer = master_agent.query(question_text)

        print('now embedding answer from OpenAI')
        encoded_answer = encode_question(answer)

        # prep metadata and upsert batch
        meta = [{'text': line}]
        to_upsert = zip(['99'], encoded_answer, meta)
        # upsert to Pinecone
        pc.index.upsert(vectors=list(to_upsert))

        return answer

# def generate_answer_with_openai(question_text):
#     # Your code to query OpenAI API and return the answer
#     return "Generated answer from OpenAI"


def main():

    index_name = "hiqa-index"
    # Initialize the DocumentProcessor and process documents
    document_processor = DocumentProcessor(llm_instance, embedding_instance)
    document_processor.load_documents()  # Ensure this method is implemented to load your documents
    # document_processor.embed_nodes_in_pinecone_index()
    document_processor.build_agents()

    # Initialize QueryManager with the DocumentProcessor instance
    query_manager = QueryManager(llm_instance, embedding_instance, document_processor)
    query_manager.build_tools()

    master_agent = query_manager.get_answer()
    query_text = "list all centers"
    query_text = "Please provide a list of Leitrim centres who have been not compliant with the fire precatuion regulation."
    # query_text = "Please provide a list of centres who have been not compliant with the fire precatuion regulation."
    response = master_agent.query(query_text)
    print(f"Response to query '{response}")


    # start_time = time.time()
    # # Submit a text query
    # query_text = "list all centers"
    # query_text = "what time is it?"
    # response = get_or_create_answer(query_manager, query_text)
    # print(response)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # minutes, seconds = divmod(elapsed_time, 60)
    # print(f"Execution time: {int(minutes)} minutes {int(seconds)} seconds")

if __name__ == "__main__":
    main()