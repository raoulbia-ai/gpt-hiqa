import asyncio
import os, json
from pathlib import Path
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

@st.cache_data
def get_wiki_titles():
    wiki_titles = []
    for file_path in data_dir_path.glob('*.pdf'):
        wiki_titles.append(file_path.stem)
    return wiki_titles

@st.cache_data
def load_documents(wiki_titles):
    # reader = UnstructuredReader()
    city_docs = {}
    for idx, wiki_title in enumerate(wiki_titles):
        try:
            # loaded_docs = reader.load_data(f"{dir_path}/{wiki_title}.pdf", split_documents=True)
            # loaded_doc = Document(
            #     text="\n\n".join([d.get_content() for d in loaded_docs]),
            #     metadata={"path": str(wiki_title)},
            # )
            # city_docs[wiki_title] = loaded_doc

            city_docs[wiki_title] = SimpleDirectoryReader(
                input_files=[f"{dir_path}/{wiki_title}.pdf"]
            ).load_data()

            print(f"Successfully loaded document: {wiki_title}")
        except Exception as e:
            print(f"Failed to load document: {wiki_title}. Error: {str(e)}")
    return city_docs

# @st.cache_data
def build_agents(wiki_titles, _city_docs):
    node_parser = SentenceSplitter()
    agents = {}
    query_engines = {}
    all_nodes = []
    for idx, wiki_title in enumerate(wiki_titles):
        # nodes = node_parser.get_nodes_from_documents([city_docs[wiki_title]])
        documents = [doc_dict[wiki_title] for doc_dict in _city_docs if wiki_title in doc_dict]
        # nodes = node_parser.get_nodes_from_documents(documents)
        flat_documents = [doc for sublist in documents for doc in sublist]
        nodes = node_parser.get_nodes_from_documents(flat_documents)
        all_nodes.extend(nodes)
        vector_index, summary_index = build_indexes(wiki_title, nodes)
        query_engine_tools = define_tools(vector_index, summary_index, wiki_title)
        agent = build_agent(query_engine_tools, wiki_title, wiki_titles)
        agents[wiki_title] = agent
        query_engines[wiki_title] = vector_index.as_query_engine(similarity_top_k=2)
    return agents, query_engines

def build_indexes(wiki_title, nodes):
    if not os.path.exists(f"{persist_path}/{wiki_title}"):
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=f"{persist_path}/{wiki_title}")
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"{persist_path}/{wiki_title}"),
        )
    summary_index = SummaryIndex(nodes)
    return vector_index, summary_index

def define_tools(vector_index, summary_index, wiki_title):

    vector_query_engine = vector_index.as_query_engine(llm=llm)
    summary_query_engine = summary_index.as_query_engine(llm=llm)
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    f"""Useful for questions related to the inspection of centre {wiki_title}"""
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    f"""Useful for any requests that require a holistic summary
                        of EVERYTHING about centre {wiki_title}. For questions about
                        more specific sections, please use the vector_tool."""
                ),
            ),
        ),
    ]
    return query_engine_tools

def build_agent(query_engine_tools, wiki_title, wiki_titles):
    system_prompt = f"""
                    You have knowledge about the following documents: {wiki_title}.
                    This document is an inspection report of disability centre {wiki_title}.
                    The first page of a document contains the following information: Name of designated centre,
                    Name of provider, Address of centre, Type of inspection, Date of inspection, and Centre ID.
                    """
    function_llm = OpenAI(model="gpt-3.5-turbo")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=system_prompt
    )
    return agent

# @st.cache_data
def define_tool_for_each_document_agent(wiki_titles, _agents):
    all_tools = []
    for wiki_title in wiki_titles:
        wiki_summary = (
            f"This content is an inspection report about {wiki_title}. Use"
            f" this tool if you want to answer any questions about {wiki_title}.\n"
        )
        doc_tool = QueryEngineTool(
            query_engine=_agents[wiki_title],
            metadata=ToolMetadata(
                name=f"{wiki_title}",
                description=wiki_summary,
            ),
        )
        all_tools.append(doc_tool)
    return all_tools

# @st.cache_data
def define_object_index_and_retriever(all_tools):
    tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
    obj_index = ObjectIndex.from_objects(
        all_tools,
        tool_mapping,
        VectorStoreIndex,
    )
    vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=10)
    return vector_node_retriever


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
                    for any queries that involve comparing multiple documents, or for queries that do NOT refer to a 
                    specific centre, use THIS tool! 
                    DO NOT USE the other tools for any queries involving multiple documents.
"""
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool", description=sub_question_description
            ),
        )

        return tools + [sub_question_tool]
    

wiki_titles = get_wiki_titles()
city_docs = load_documents(wiki_titles)
agents, query_engines = build_agents(wiki_titles, [city_docs])
all_tools = define_tool_for_each_document_agent(wiki_titles, agents)
vector_node_retriever = define_object_index_and_retriever(all_tools)

custom_node_retriever = CustomRetriever(vector_node_retriever)

# wrap it with ObjectRetriever to return objects
tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
obj_index = ObjectIndex.from_objects(
    all_tools,
    tool_mapping,
    VectorStoreIndex,
)

custom_obj_retriever = CustomObjectRetriever(
    custom_node_retriever, tool_mapping, all_tools, llm=llm
)



top_agent = FnRetrieverOpenAIAgent.from_retriever(
    custom_obj_retriever,
    system_prompt=f""" 
You are an AI expert in disability centre inspections, with a specialized focus on "The Health 
Information and Quality Authority" (HIQA). HIQA is an independent authority established to drive 
high-quality and safe care for people using our health and social care services in Ireland. HIQA’s 
mandate to date extends across a specified range of public, private and voluntary sector services. 

You have knowledge about the following documents: {wiki_titles}.
The documents contain details about inspections of disability centre {wiki_titles}.
                                                        
Do NOT rely on prior or external knowledge.

If a query or question does not reference a specific center by name then your response should integrate
document information of ALL documents / reports that you have been trained on. 
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


async def handle_input(conversation):
    user_input = st.session_state.question_input
    if user_input:
        # Add question to conversation
        conversation.append(("You", user_input))

        prompt = ''
        response = await top_agent.query(user_input)
        # print(response)
        answer = get_response_without_metadata(response)

        # Save the question, the top answer, and the timestamp to a CSV file
        # with open('questions_answers.csv', 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     # Write the question, the top answer, and the timestamp to the CSV file
        #     # Assuming reranked_results[0] is the top answer
        #     writer.writerow([user_input, answer])

        # Add answer to conversation
        conversation.append(("AI", answer))
        # Clear input box
        st.session_state.question_input = ""


if __name__ == "__main__":
    asyncio.run(main())
    main()