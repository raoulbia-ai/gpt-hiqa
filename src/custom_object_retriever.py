from config import COHERE_API_KEY
from llama_index.core.retrievers import BaseRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.objects import ObjectRetriever
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
# from src.document_processor import DocumentProcessor
# from src.query_manager import QueryManager


class CustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever, all_tools, postprocessor=None):
        self._vector_retriever = vector_retriever
        self._postprocessor = postprocessor or CohereRerank(top_n=5)
        self.all_tools = all_tools
        super().__init__()

    def _retrieve(self, query_bundle):
        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        filtered_nodes = self._postprocessor.postprocess_nodes(
            retrieved_nodes, query_bundle=query_bundle
        )

        return filtered_nodes


class CustomObjectRetriever(ObjectRetriever):
    def __init__(self,
                 retriever,
                 object_node_mapping,
                 all_tools,
                 llm=None):
        super().__init__(retriever, object_node_mapping)
        self._llm = llm
        self.all_tools = all_tools

    def retrieve(self, query_bundle):
        nodes = self._retriever.retrieve(query_bundle)
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=self.all_tools, #tools,
            llm=self._llm
        )
        sub_question_description = f"""\
                Useful for any queries that involve comparing multiple documents. 
                ALWAYS use this tool for comparison queries - make sure to call this \
                tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
                RETURN YOUR RESPONSE AS SOON AS POSSIBLE.
                """
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool",
                description=sub_question_description
            ),
        )

        return tools + [sub_question_tool]