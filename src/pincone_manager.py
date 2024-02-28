from pinecone import Pinecone, ServerlessSpec, PodSpec
from typing import List, Dict
from config import PINECONE_API_KEY, PINECONE_ENV

# PINECONE_ENV = 'us-west-2'


class PineconeManager:
    def __init__(self, index_name: str, dimension: int = 1536, metric: str = 'cosine', cloud: str = 'aws',
                 region: str = PINECONE_ENV):

        # Initialize a Pinecone instance with your API key
        # super().__init__(**kwargs)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        # self.index = self.pc.Index(index_name)

        # Check if the index exists, and create it if it does not
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                # spec=ServerlessSpec(cloud=cloud, region=region)
                spec=PodSpec(
                    environment=PINECONE_ENV, #"us-west1-gcp",
                    pod_type="p1.x1",
                    pods=1)
            )

        # Connect to the index
        self.index = self.pc.Index(name=index_name)

    def upsert(self, vectors: List[Dict]):
        """
        Insert or update vectors in the index.

        :param vectors: A list of dictionaries with 'id' and 'values' as keys.
        """
        self.index.upsert(items=vectors)

    def query(self, vector: List[float], top_k: int = 5):
        """
        Query the index for similar vectors.

        :param vector: The query vector.
        :param top_k: Number of closest vectors to return.
        :return: Query results.
        """
        return self.index.query(queries=[{"vector": vector}], top_k=top_k)

    def delete(self, vector_ids: List[str]):
        """
        Delete vectors from the index by their IDs.

        :param vector_ids: A list of vector IDs to delete.
        """
        self.index.delete(ids=vector_ids)

    def list_indeces(self):
        return self.pc.list_indexes()


# Example usage
if __name__ == "__main__":
    pass
    # INDEX_NAME = "my-test-index"

    # Instantiate the VectorIndex
    # vector_index = PineconeManager(INDEX_NAME)

    # Example vector to upsert
    # example_vectors = [{"id": "vector1", "values": [0.1, 0.2, 0.3, ..., 0.768]}]  # Adjust the dimension as necessary

    # vector_index.upsert(vectors=example_vectors)
    #
    # # Query the index
    # query_result = vector_index.query(vector=[0.1, 0.2, 0.3, ..., 0.768], top_k=5)
    # print(query_result)
    #
    # # Delete a vector
    # vector_index.delete(vector_ids=["vector1"])
