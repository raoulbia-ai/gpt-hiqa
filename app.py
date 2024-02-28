import streamlit as st
from src.document_processor import DocumentProcessor  # Adjust import path as necessary
from src.query_manager import QueryManager  # Adjust import path as necessary
from config import llm_instance, embedding_instance

# Initialize session state once
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Function to get the session state
def get_session_state():
    return st.session_state


# Function to set the session state
def set_session_state(**kwargs):
    for key, value in kwargs.items():
        st.session_state[key] = value

def initialize():
    index_name = "hiqa-index"
    if 'initialized' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor(llm_instance, embedding_instance, index_name)
        st.session_state.doc_processor.load_documents()
        st.session_state.doc_processor.embed_nodes_in_pinecone_index()
        st.session_state.doc_processor.build_agents()

        # Initialize the QueryManager with the DocumentProcessor instance
        st.session_state.query_manager = QueryManager(llm_instance, embedding_instance, st.session_state.doc_processor)
        st.session_state.query_manager.build_tools()
        # Mark as initialized to prevent re-initialization in the same session
        st.session_state.initialized = True

def main():
    st.markdown("# HIQA Inspection Reports Q&A")
    st.markdown("""Proof of Concept ChatGPT Application trained on inspection reports for 
        disability centers in Leitrim.""")
    st.markdown("""To see a list of available documents, simply type: list""")

    initialize()

    # Input for questions
    user_input = st.text_input("Enter your question:", key='question_input')

    if st.button('➡️'):
        handle_input(st.session_state.conversation, user_input)

    # Display conversation
    for speaker, text in st.session_state.conversation:
        st.write(f"{speaker}: {text}")


def handle_input(conversation, user_input):
    if user_input:
        # Add question to conversation
        conversation.append(("You", user_input))

        st.session_state['processing'] = True

        with st.spinner('Processing...'):
            response = st.session_state['query_manager'].get_answer().query(user_input)

        # Add answer to conversation
        conversation.append(("AI", response))
        # Clear input box
        st.session_state['processing'] = False

if __name__ == '__main__':
    main()