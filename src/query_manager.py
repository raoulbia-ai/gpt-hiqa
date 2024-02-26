from src.custom_object_retriever import CustomObjectRetriever, CustomRetriever
from src.document_processor import DocumentProcessor
from llama_index.agent.openai_legacy import FnRetrieverOpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core import VectorStoreIndex
import time


class QueryManager:
    def __init__(self, llm, embed_model, document_processor):
        self.document_processor = document_processor
        self.llm = llm
        self.embed_model = embed_model
        self.all_tools = []
        self._cached_master_agent = None
        self.cache_used = False


    def build_tools(self):
        for file_base, agent in self.document_processor.agents_dict.items():
            summary = self.document_processor.extra_info_dict[file_base]["summary"]
            doc_tool = QueryEngineTool(
                query_engine=agent,
                metadata=ToolMetadata(
                    name=f"{file_base}",
                    description='',
                ),
            )
            self.all_tools.append(doc_tool)

    def get_answer(self):
        start_time = time.time()
        if not self._cached_master_agent:
            self.cache_used = False
            tool_mapping = SimpleToolNodeMapping.from_objects(self.all_tools)
            obj_index = ObjectIndex.from_objects(
                self.all_tools,
                tool_mapping,
                VectorStoreIndex,
            )
            vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=10)

            custom_node_retriever = CustomRetriever(vector_node_retriever)

            custom_obj_retriever = CustomObjectRetriever(
                custom_node_retriever, tool_mapping, self.all_tools, llm=self.llm
            )
            self._cached_master_agent = FnRetrieverOpenAIAgent.from_retriever(
                custom_obj_retriever,
                system_prompt=f"""\ 
                        You are an AI expert in disability centre inspections, with a specialized focus on "The Health \
                        Information and Quality Authority" (HIQA). HIQA is an independent authority established to drive \ 
                        high-quality and safe care for people using our health and social care services in Ireland. HIQA’s \ 
                        mandate to date extends across a specified range of public, private and voluntary sector services. \ 
                        You have knowledge about the following documents: \
                        {self.document_processor.titles} \
                        These documents are inspection reports of disability centres. \ 
                        Reports may cover inspections at the same centre at different dates. \ 
                        You must ALWAYS use at least one of the tools provided when answering a question.g \
                        Do NOT rely on prior or external knowledge. \
                        If a user input is `/list`, then list the names of the centres, their addresses and the date or \
                         dates each centre has been inspected. Group dates by centre. \
                        For each center, list the center name followed by the dates of inspections scheduled there, \
                        sorted chronologically. Ensure the information is presented clearly and concisely for easy \
                        understanding. \
                        Response Format should be a table with two columns: \ 
                        - Centre Name \
                        - Inspection Date(s) \
                        """,
                llm=self.llm,
                verbose=True,
            )
        else:
            self.cache_used = True
        end_time = time.time()
        print(f"Cache used: {self.cache_used}")
        print(f"Time taken: {end_time - start_time} seconds")
        return self._cached_master_agent
