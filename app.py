import streamlit as st
from src.document_processor import DocumentProcessor  # Adjust import path as necessary
from src.query_manager import QueryManager  # Adjust import path as necessary
from config import llm_instance, embedding_instance
import time

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
    if 'initialized' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor(llm_instance, embedding_instance)
        st.session_state.doc_processor.load_documents()
        # Assuming build_agents and build_tools are methods of DocumentProcessor
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
    st.markdown("""To see a list of available documents, simply type: /list""")

    initialize()

    # Input for questions
    user_input = st.text_input("Enter your question:", key='question_input')

    if st.button('➡️'):
        time_taken = handle_input(st.session_state.conversation, user_input)

    # Display conversation
    for speaker, text in st.session_state.conversation:
        st.write(f"**Cache used:** {cache_used}")
        st.write(f"**Time taken:** {time_taken:.2f} seconds")
        st.write(f"{speaker}: {text}")



def handle_input(conversation, user_input):
    if user_input:
        # Add question to conversation
        conversation.append(("You", user_input))

        st.session_state['processing'] = True

        start_time = time.time()
        with st.spinner('Processing...'):
            response = st.session_state['query_manager'].get_answer().query(user_input)
        end_time = time.time()

        # Add answer to conversation
        conversation.append(("AI", response))

        # Display timing and cache usage information separately
        # Calculate timing and cache usage information
        cache_used = st.session_state['query_manager'].cache_used
        time_taken, cache_used = end_time - start_time

        # Add answer and timing information to conversation
        # response_with_timing = f"{response}\n(Cache used: {cache_used}, Time taken: {time_taken:.2f} seconds)"
        # conversation.append(("AI", response_with_timing))

        # Add answer to conversation
        conversation.append(("AI", response))
        # Display timing and cache usage information
        cache_used = st.session_state['query_manager'].cache_used
        time_taken = end_time - start_time

        # Clear input box
        st.session_state['processing'] = False
        # Reset cache_used attribute for the next query
        st.session_state['query_manager'].cache_used = False

        return time_taken, cache_used


if __name__ == '__main__':
    main()
