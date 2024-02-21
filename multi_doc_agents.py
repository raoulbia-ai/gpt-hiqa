import nest_asyncio
nest_asyncio.apply()

import os, json
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

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

dir_path = 'data/hiqa_pdfs'
persist_path = 'persist'
data_dir_path = Path(dir_path)

llm = OpenAI(temperature=0, model='gpt-3.5-turbo')

titles = []
for file_path in dir_path.glob('*.pdf'):
    titles.append(file_path.stem)


@st.cache_resource
def load_documents(titles):
    docs = []
    for idx, title in enumerate(titles):
        try:
            docs.append(SimpleDirectoryReader(
                input_files=[f"{dir_path}/{title}.pdf"]
            ).load_data())
            print(f"Successfully loaded document: {title}")
        except Exception as e:
            print(f"Error loading document: {title}. Error: {str(e)}")
    return docs

st.session_state.docs_loaded = False
if 'docs_loaded' not in st.session_state:
    st.session_state.docs = load_documents(titles)
    st.session_state.docs_loaded = True
docs = st.session_state.city_docs


Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

async def build_agent_per_doc(nodes, file_base):
    print(file_base)

    vi_out_path = f"./data/llamaindex_docs/{file_base}"
    summary_out_path = f"./data/llamaindex_docs/{file_base}_summary.pkl"
    if not os.path.exists(vi_out_path):
        Path("./data/llamaindex_docs/").mkdir(parents=True, exist_ok=True)
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
    function_llm = OpenAI(model="gpt-4")
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

    # # this is for the baseline
    # all_nodes = []

    for idx, doc in enumerate(tqdm(docs)):
        nodes = node_parser.get_nodes_from_documents([doc])
        # all_nodes.extend(nodes)

        # ID will be base + parent
        file_path = Path(doc.metadata["path"])
        file_base = str(file_path.parent.stem) + "_" + str(file_path.stem)
        agent, summary = await build_agent_per_doc(nodes, file_base)

        agents_dict[file_base] = agent
        extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}

    return agents_dict, extra_info_dict


agents_dict, extra_info_dict = await build_agents(docs)


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
    if 'docs_loaded' not in st.session_state:
        st.session_state.docs = load_documents(titles)
        st.session_state.docs_loaded = True


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

        if 'processing' in st.session_state and not st.session_state['processing']:
            # Add answer to conversation
            conversation.append(("AI", answer))
            # Clear input box
            st.session_state['processing'] = False
            st.session_state.question_input = ""


# if __name__ == "__main__":
#     main()
