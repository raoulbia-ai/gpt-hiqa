from config import COHERE_API_KEY
from llama_index.core.retrievers import BaseRetriever
from functools import lru_cache
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.objects import ObjectRetriever
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

from functools import lru_cache

def hashable_query_bundle(query_bundle):
    # Convert the query_bundle into a hashable representation
    # This is a placeholder function and should be adapted to the structure of QueryBundle
    return str(query_bundle)

class CustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever, postprocessor=None):
        self._vector_retriever = vector_retriever
        self._postprocessor = postprocessor or CohereRerank(top_n=5)
        super().__init__()

    def _retrieve_with_cache(self, hashable_query):
        # Convert the hashable query back to the original query_bundle
        # This is a placeholder function and should be adapted to the structure of QueryBundle
        query_bundle = eval(hashable_query)
        return self._retrieve(query_bundle)

    _retrieve_with_cache = lru_cache(maxsize=128)(_retrieve_with_cache)

    def _retrieve(self, query_bundle):
        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        filtered_nodes = self._postprocessor.postprocess_nodes(
            retrieved_nodes, query_bundle=query_bundle
        )

        return filtered_nodes


class CustomObjectRetriever(ObjectRetriever):
    def __init__(self, retriever, object_node_mapping, all_tools, llm=None):
        super().__init__(retriever, object_node_mapping)
        self._llm = llm  # Add your logic for llm initialization

    def retrieve(self, query_bundle):
        hashable_query = hashable_query_bundle(query_bundle)
        nodes = self._retriever._retrieve_with_cache(hashable_query)
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools, llm=self._llm
        )
        sub_question_description = f"""\
                Useful for any queries that involve comparing multiple documents. 
                ALWAYS use this tool for comparison queries - make sure to call this \
                tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
                """
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool",
                description=sub_question_description
            ),
        )

        return tools + [sub_question_tool]
