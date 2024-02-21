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


wiki_titles = []
for file_path in data_dir_path.glob('*.pdf'):
    wiki_titles.append(file_path.stem)


city_docs = {}
for idx, wiki_title in enumerate(wiki_titles):
    try:

        city_docs[wiki_title] = SimpleDirectoryReader(
            input_files=[f"{dir_path}/{wiki_title}.pdf"]
        ).load_data()

        print(f"Successfully loaded document: {wiki_title}")
    except Exception as e:
        print(f"Error creating sub-question tool 'compare_tool': {e}")
        print(f"Failed to load document: {wiki_title}. Error: {str(e)}")


node_parser = SentenceSplitter()

# Build agents dictionary
agents = {}
query_engines = {}

# this is for the baseline
all_nodes = []

for idx, wiki_title in enumerate(wiki_titles):
    nodes = node_parser.get_nodes_from_documents(city_docs[wiki_title])
    all_nodes.extend(nodes)

    if not os.path.exists(f"./data/{wiki_title}"):
        # build vector index
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(
            persist_dir=f"./data/{wiki_title}"
        )
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./data/{wiki_title}"),
        )

    # build summary index
    summary_index = SummaryIndex(nodes)
    # define query engines
    vector_query_engine = vector_index.as_query_engine(llm=llm)
    summary_query_engine = summary_index.as_query_engine(llm=llm)

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    "Useful for questions related to specific aspects of"
                    f" {wiki_title} (e.g. the history, arts and culture,"
                    " sports, demographics, or more)."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    "Useful for any requests that require a holistic summary"
                    f" of EVERYTHING about {wiki_title}. For questions about"
                    " more specific sections, please use the vector_tool."
                ),
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
You are a specialized agent designed to answer queries about {wiki_title}.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )

    agents[wiki_title] = agent
    query_engines[wiki_title] = vector_index.as_query_engine(
        similarity_top_k=2
    )

# define tool for each document agent
all_tools = []
for wiki_title in wiki_titles:
    wiki_summary = (
        f"This content contains Wikipedia articles about {wiki_title}. Use"
        f" this tool if you want to answer any questions about {wiki_title}.\n"
    )
    doc_tool = QueryEngineTool(
        query_engine=agents[wiki_title],
        metadata=ToolMetadata(
            name=f"tool_{wiki_title}",
            description=wiki_summary,
        ),
    )
    all_tools.append(doc_tool)

llm = OpenAI(temperature=0, model='gpt-4-0613')

tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
obj_index = ObjectIndex.from_objects(
    all_tools,
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

custom_obj_retriever = CustomObjectRetriever(
    custom_node_retriever, tool_mapping, all_tools, llm=llm
)

top_agent = FnRetrieverOpenAIAgent.from_retriever(
    custom_obj_retriever,
    system_prompt=""" 
You are an AI expert in disability centre inspections, with a specialized focus on "The Health 
                    Information and Quality Authority" (HIQA). HIQA is an independent authority established to drive 
                    high-quality and safe care for people using our health and social care services in Ireland. HIQA’s 
                    mandate to date extends across a specified range of public, private and voluntary sector services. 

                    You have knowledge about the following documents:

                    {wiki_titles}

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


if __name__ == "__main__":
    main()
