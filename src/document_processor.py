import os
import pickle
from pathlib import Path
from config import OPENAI_API_KEY
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, SummaryIndex
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.agent.openai import OpenAIAgent
# from llama_index.llms.azure_openai import AzureOpenAI

class DocumentProcessor:
    def __init__(self, llm, embed_model, documents_dir='data/hiqa_pdfs'):
        self.documents_dir = documents_dir
        self.documents = []
        self.agents_dict = {}
        self.extra_info_dict = {}
        # Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
        # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        self.llm = llm
        self.embed_model = embed_model
        # self.llm = OpenAI(temperature=0, model='gpt-4')
        # self.model = 'gpt-3.5-turbo'
        data_dir_path = Path(self.documents_dir)
        self.titles = [file_path.stem for file_path in data_dir_path.glob('*.pdf')]

    def load_documents(self):

        for title in self.titles:
            try:
                doc = SimpleDirectoryReader(
                    input_files=[f"{self.documents_dir}/{title}.pdf"]
                ).load_data()

                loaded_doc = Document(
                    text="\n\n".join(d.get_content() for d in doc),
                    metadata={"path": str(title)}
                )
                self.documents.append(loaded_doc)
                print(f"Successfully loaded document: {title}")
            except Exception as e:
                print(f"Error loading document: {title}. Error: {str(e)}")

    def extract_text_from_pdf(self, file_path):
        # Existing extract_text_from_pdf implementation
        pass

    def build_agent_per_doc(self, nodes, file_base):
        # print(f'file_base: {file_base}')

        vi_out_path = f"persist/{file_base}"
        summary_out_path = f"persist/{file_base}_summary.pkl"
        if not os.path.exists(vi_out_path):
            Path("persist/").mkdir(parents=True, exist_ok=True)
            # build vector index
            vector_index = VectorStoreIndex(nodes)
            vector_index.storage_context.persist(persist_dir=vi_out_path)
        else:
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=vi_out_path),
            )

        # build summary index
        summary_index = SummaryIndex(nodes)

        # define query engines
        vector_query_engine = vector_index.as_query_engine(llm=self.llm)
        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize", llm=self.llm
        )

        # extract a summary
        if not os.path.exists(summary_out_path):
            Path(summary_out_path).parent.mkdir(parents=True, exist_ok=True)
            summary = str(
                summary_query_engine.aquery(
                    "Extract a concise 1-2 sentence summary of this document."
                )
            )
            pickle.dump(summary, open(summary_out_path, "wb"))
        else:
            summary = pickle.load(open(summary_out_path, "rb"))

        # define tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name=f"{file_base}",
                    description=f"Useful for questions related to specific facts about {file_base} ",
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name=f"{file_base}",
                    description=f"Useful for summarization questions",
                ),
            ),
        ]

        # build agent
        # function_llm = OpenAI(model=self.model)
        agent = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=self.llm, #function_llm,
            verbose=True,
            system_prompt=f"""
                                You are a specialized agent designed to answer queries about the {file_base} document.
                                You must ALWAYS use at least one of the tools provided when answering a question; 
                                do NOT rely on prior knowledge.\
                                """,
        )

        return agent, summary

    def build_agents(self):
        node_parser = SentenceSplitter()
        for doc in self.documents:
            nodes = node_parser.get_nodes_from_documents([doc])
            file_path = Path(doc.metadata["path"])
            file_base = str(file_path.stem)
            agent, summary = self.build_agent_per_doc(nodes, file_base)

            self.agents_dict[file_base] = agent
            self.extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}

# Assuming the rest of the functionality is implemented elsewhere or as needed
