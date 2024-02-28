import os
import json
import pickle
from pathlib import Path
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, SummaryIndex
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import Settings
# from llama_index.llms.openai import OpenAI
from openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.agent.openai import OpenAIAgent
from config import HF_TOKEN
from src.pincone_manager import PineconeManager
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
# from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.pinecone import PineconeReader
from llama_index.vector_stores.pinecone import PineconeVectorStore

class DocumentProcessor:
    def __init__(self, llm, embed_model, index_name, documents_dir='data/hiqa_pdfs'):
        self.index_name = index_name
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

        self.client = OpenAI()
        # text-embedding-ada-002 produces vectors with 1536 dimensions
        self.EMBED_MODEL = "text-embedding-ada-002"

        # we need an embedding of dim 768 hence we use all-mpnet-base-v2
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", use_auth_token=HF_TOKEN)
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", use_auth_token=HF_TOKEN)
        self.pc = PineconeManager(index_name=self.index_name)
        self.pc_vector_store = PineconeVectorStore(api_key=PINECONE_API_KEY,
                                                   environment=PINECONE_ENV,
                                                   index_name=self.index_name)
        self.id_to_text_map = {}

        self.node_parser = SentenceSplitter(
                                            # chunk_size=1024,
                                            # chunk_overlap=20
                                            )
        # self.nodes = None

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

    def build_agent_per_docOLD(self, nodes, file_base):
        # print(f'file_base: {file_base}')

        # persist or load vector index
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
        # https://colab.research.google.com/drive/1ZAdrabTJmZ_etDp10rjij_zME2Q3umAQ?usp=sharing#scrollTo=aCdR2_wmNol6
        vector_query_engine = vector_index.as_query_engine(
            response_mode="compact",
            llm=self.llm)
        summary_query_engine = summary_index.as_query_engine(
            response_mode="compact", #"tree_summarize",     # <<<<<<<<<<<<<<<<  look up difference (see also "refine")
            llm=self.llm
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

    def build_agentsOLD(self):
        node_parser = SentenceSplitter(
                                        # chunk_size=1024,
                                        # chunk_overlap=20
                                        )
        for doc in self.documents:
            nodes = node_parser.get_nodes_from_documents([doc])

            self.embed_nodes_in_pinecone_index(nodes)  # <<<<<<<<<<<< NEW

            file_path = Path(doc.metadata["path"])
            file_base = str(file_path.stem)
            agent, summary = self.build_openai_agent_per_doc_from_pinecone(nodes, file_base)

            self.agents_dict[file_base] = agent
            self.extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}

    def build_openai_agent_per_doc_from_pinecone(self, nodes, file_base):
        print(f'building OpenAI agent (with query engines) for: {file_base}')

        # PINECONE
        vector_store = self.pc_vector_store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=storage_context
        )

        # build summary index
        summary_index = SummaryIndex(nodes)

        # define query engines
        vector_query_engine = vector_index.as_query_engine(
            response_mode="compact",
            llm=self.llm)
        summary_query_engine = summary_index.as_query_engine(
            response_mode="compact",  # "tree_summarize",     # <<<<<<<<<<<<<<<<  look up difference (see also "refine")
            llm=self.llm
        )

        # # extract a summary
        # if not os.path.exists(summary_out_path):
        #     Path(summary_out_path).parent.mkdir(parents=True, exist_ok=True)
        #     summary = str(
        #         summary_query_engine.aquery(
        #             "Extract a concise 1-2 sentence summary of this document."
        #         )
        #     )
        #     pickle.dump(summary, open(summary_out_path, "wb"))
        # else:
        #     summary = pickle.load(open(summary_out_path, "rb"))

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

        return agent #, summary

    def build_agents(self):
        for doc in self.documents:
            nodes = self.node_parser.get_nodes_from_documents([doc])
            file_path = Path(doc.metadata["path"])
            print(f'Path(doc.metadata["path"]): {file_path}')
            file_base = str(file_path.stem)
            agent = self.build_openai_agent_per_doc_from_pinecone(nodes, file_base)

            self.agents_dict[file_base] = agent

    def embed_nodes_in_pinecone_index(self):
        if self.index_name not in self.pc.list_indeces().names():
            print(f"Starting to build Pinecone index {self.index_name}!")

            # Instantiate the Pinecone VectorIndex
            PineconeManager(index_name=self.index_name)

            for doc in self.documents:
                nodes = self.node_parser.get_nodes_from_documents([doc])
                for node in nodes:
                    self.id_to_text_map[node.id_] = node.text
                    embedding = self.client.embeddings.create(
                        input=node.text, model=self.EMBED_MODEL
                    )
                    # print(embedding)
                    embedding = [record.embedding for record in embedding.data]

                    # prep metadata and upsert batch
                    meta = [{'text': node.text}]
                    to_upsert = zip([f'{node.id_}'], embedding, meta)
                    # upsert to Pinecone
                    self.pc.index.upsert(vectors=list(to_upsert))
            print(f"Done building Pinecone index {self.index_name}!")
        else:
            print(f"Pinecone index {self.index_name} already exists. Moving on to querying....")


