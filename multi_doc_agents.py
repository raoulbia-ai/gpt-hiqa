import nest_asyncio
nest_asyncio.apply()
import asyncio
from tqdm import tqdm
import os, json, pickle
from pathlib import Path
# from streamlit import caching
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
)
from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from llama_index.agent.openai import OpenAIAgent
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter

# from llama_index.readers.file import UnstructuredReader
from llama_index.core.schema import Document
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.objects import ObjectRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import SubQuestionQueryEngine

from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.agent.openai_legacy import FnRetrieverOpenAIAgent

from llama_index.core import Document

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

dir_path = 'data/hiqa_pdfs'
persist_path = 'persist'
data_dir_path = Path(dir_path)

llm = OpenAI(temperature=0, model='gpt-3.5-turbo')

titles = []
for file_path in data_dir_path.glob('*.pdf'):
    titles.append(file_path.stem)


@st.cache_resource
def load_documents(titles):
    docs = []
    for idx, d in enumerate(titles):
        try:
            doc = SimpleDirectoryReader(
                input_files=[f"{dir_path}/{d}.pdf"]
            ).load_data()

            loaded_doc = Document(
                text="\n\n".join([d.get_content() for d in doc]),
                metadata={"path": str(d)},
            )
            # print(loaded_doc.metadata["path"])
            docs.append(loaded_doc)

            print(f"Successfully loaded document: {d}")
        except Exception as e:
            print(f"Error loading document: {d}. Error: {str(e)}")
    return docs


Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

async def build_agent_per_doc(nodes, file_base):
    print(file_base)

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
    vector_query_engine = vector_index.as_query_engine(llm=llm)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize", llm=llm
    )

    # extract a summary
    if not os.path.exists(summary_out_path):
        Path(summary_out_path).parent.mkdir(parents=True, exist_ok=True)
        summary = str(
            await summary_query_engine.aquery(
                "Extract a concise 1-2 line summary of this document"
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
                name=f"vector_tool_{file_base}",
                description=f"Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{file_base}",
                description=f"Useful for summarization questions",
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-3.5-turbo")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\
You are a specialized agent designed to answer queries about the `{file_base}.html` part of the LlamaIndex docs.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )

    return agent, summary



async def build_agents(docs):
    node_parser = SentenceSplitter()

    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    if 'agents_dict' not in st.session_state:
            st.session_state['agents_dict'] = {}
            st.session_state['extra_info_dict'] = {}
   
    for idx, doc in enumerate(tqdm(docs)):
        nodes = node_parser.get_nodes_from_documents([doc])
        # all_nodes.extend(nodes)

        # ID will be base + parent
        file_path = Path(doc.metadata["path"])
        file_base = str(file_path.parent.stem) + "_" + str(file_path.stem)
        agent, summary = await build_agent_per_doc(nodes, file_base)

        st.session_state['agents_dict'][file_base] = agent
        st.session_state['extra_info_dict'][file_base] = {"summary": summary, "nodes": nodes}

    return st.session_state['agents_dict'], st.session_state['extra_info_dict']

docs = load_documents(titles)
agents_dict, extra_info_dict = asyncio.run(build_agents(docs))


if 'all_tools' not in st.session_state:
    # define tool for each document agent
    st.session_state['all_tools'] = []
    for file_base, agent in agents_dict.items():
        summary = extra_info_dict[file_base]["summary"]
        doc_tool = QueryEngineTool(
            query_engine=agent,
            metadata=ToolMetadata(
                name=f"tool_{file_base}",
                description=summary,
            ),
        )
        st.session_state['all_tools'].append(doc_tool)

llm = OpenAI(model_name="gpt-3.5-turbo")

tool_mapping = SimpleToolNodeMapping.from_objects(st.session_state['all_tools'])
obj_index = ObjectIndex.from_objects(
    st.session_state['all_tools'],
    tool_mapping,
    VectorStoreIndex,
)
vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=10)

# define a custom retriever with reranking
class CustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever, postprocessor=None):
        self._vector_retriever = vector_retriever
        self._postprocessor = postprocessor or CohereRerank(top_n=5)
        super().__init__()

    def _retrieve(self, query_bundle):
        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        filtered_nodes = self._postprocessor.postprocess_nodes(
            retrieved_nodes, query_bundle=query_bundle
        )

        return filtered_nodes


# define a custom object retriever that adds in a query planning tool
class CustomObjectRetriever(ObjectRetriever):
    def __init__(self, retriever, object_node_mapping, all_tools, llm=None):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._llm = llm or OpenAI("gpt-3.5-turbo")

    def retrieve(self, query_bundle):
        nodes = self._retriever.retrieve(query_bundle)
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools, llm=self._llm
        )
        sub_question_description = f"""\
Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
"""
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool", description=sub_question_description
            ),
        )

        return tools + [sub_question_tool]


custom_node_retriever = CustomRetriever(vector_node_retriever)

# wrap it with ObjectRetriever to return objects
custom_obj_retriever = CustomObjectRetriever(
    custom_node_retriever, tool_mapping, st.session_state['all_tools'], llm=llm
)

top_agent = FnRetrieverOpenAIAgent.from_retriever(
    custom_obj_retriever,
    system_prompt=f""" 
                    You are an AI expert in disability centre inspections, with a specialized focus on "The Health 
                    Information and Quality Authority" (HIQA). HIQA is an independent authority established to drive 
                    high-quality and safe care for people using our health and social care services in Ireland. HIQA’s 
                    mandate to date extends across a specified range of public, private and voluntary sector services. 

                    You have knowledge about the following documents:
                    
                    {titles}
                    
                    The first page of a document contains the following information:
                        - Name of designated centre
                        - Name of provider
                        - Address of centre
                        - Type of inspection
                        - Date of inspection
                        - Centre ID
                        
                    The document sections are:
                        - About the designated centre
                        - Number of residents on date of inspection
                        - How we inspect
                        - Date, Times of inspection, Inspector, Role
                        - What residents told us and what inspectors observed
                        - Capacity and capability
                        - Several sections related to specific regulations and their corresponding inspection outcome (aka judgement)
                        - Quality and safety
                        - Appendix 1 - Full list of regulations considered under each dimension
                        - Compliance Plan for the inspected centre
                        - Compliance plan provider’s response
                        - Summary of regulations to be complied with incl. Risk Rating and date to be complied with
                        
                        

                    These documents are inspection reports of disability centres. 
                    Reports may cover inspections at the same centre at different dates. 

                    Ensure your responses are comprehensive and tailored for an audience knowledgeable 
                    in the field. 

                    You must ALWAYS use at least one of the tools provided when answering a question.
                    
                    If a question is not specific to a particular centre, you MUST include ALL
                    centres in your response! 

                    Do NOT rely on prior knowledge.
""",
    llm=llm,
    verbose=True,
)


# Function to get the session state
def get_session_state():
    return st.session_state


# Function to set the session state
def set_session_state(**kwargs):
    for key, value in kwargs.items():
        st.session_state[key] = value


# Function to get the response without metadata
def get_response_without_metadata(response):
    # print(type(response))
    # print(response)
    return response  # response['choices'][0]['text']


def main():
    # Clear cache if needed
    # caching.clear_cache()

    # Load documents if not already loaded
    # This block is redundant as the loading is handled above and should be removed


    if 'agents_dict' not in st.session_state or 'extra_info_dict' not in st.session_state:
        st.session_state['agents_dict'], st.session_state['extra_info_dict'] = asyncio.run(build_agents(docs))

    try:
        st.title("HIQA Inspection Reports Q&A")
        st.write("""Proof of Concept ChatGPT Application trained on inspection reports for 
            disability centers in Leitrim.""")

        # Session state to store conversation history
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []

        # Input for questions
        user_input = st.text_input("Enter your question:", key='question_input', on_change=handle_input,
                                   args=(st.session_state.conversation,))

        # Display conversation
        for speaker, text in st.session_state.conversation:
            st.write(f"{speaker}: {text}")

        # Use the cached agents and extra info from the session state
        agents_dict = st.session_state['agents_dict']
        extra_info_dict = st.session_state['extra_info_dict']

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

def handle_input(conversation):
    user_input = st.session_state.question_input
    if user_input:
        # Add question to conversation
        conversation.append(("You", user_input))

        st.session_state['processing'] = True

        prompt = ''
        response = top_agent.query(user_input)
        # print(response)
        answer = get_response_without_metadata(response)

        # if 'processing' in st.session_state and not st.session_state['processing']:
        # Add answer to conversation
        conversation.append(("AI", answer))
        # Clear input box
        st.session_state['processing'] = False
        st.session_state.question_input = ""


if __name__ == "__main__":
    main()
