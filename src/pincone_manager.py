from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
from config import PINECONE_API_KEY, PINECONE_ENV

from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
from config import PINECONE_API_KEY, PINECONE_ENV  # Assuming these are defined in your config.py


class VectorIndex:
    def __init__(self, index_name: str, dimension: int = 768, metric: str = 'cosine', cloud: str = 'aws',
                 region: str = 'us-west-2'):
        """
        Initialize the Pinecone Vector Index using the updated Pinecone API.

        :param index_name: Name of the Pinecone index.
        :param dimension: Dimension of the vectors.
        :param metric: Metric to use for the index.
        :param cloud: Cloud provider for the Pinecone service.
        :param region: Cloud region for the Pinecone service.
        """
        # Initialize a Pinecone instance with your API key
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # Check if the index exists, and create it if it does not
        if index_name not in self.pc.list_indexes().names:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
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


# Example usage
if __name__ == "__main__":
    INDEX_NAME = "your_index_name"

    # Instantiate the VectorIndex
    vector_index = VectorIndex(index_name=INDEX_NAME)

    # # Example vector to upsert
    # example_vectors = [{"id": "vector1", "values": [0.1, 0.2, 0.3, ..., 0.768]}]  # Adjust the dimension as necessary
    # vector_index.upsert(vectors=example_vectors)
    #
    # # Query the index
    # query_result = vector_index.query(vector=[0.1, 0.2, 0.3, ..., 0.768], top_k=5)
    # print(query_result)
    #
    # # Delete a vector
    # vector_index.delete(vector_ids=["vector1"])
