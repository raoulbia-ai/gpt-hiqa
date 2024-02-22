from src.document_processor import DocumentProcessor
from src.query_manager import QueryManager



def main():
    # Initialize the DocumentProcessor and process documents
    document_processor = DocumentProcessor()
    document_processor.load_documents()  # Ensure this method is implemented to load your documents
    document_processor.build_agents()

    # Initialize QueryManager with the DocumentProcessor instance
    query_manager = QueryManager(document_processor)
    query_manager.build_tools()
    master_agent = query_manager.get_answer()

    # Submit a text query
    query_text = "list all centers"
    response = master_agent.query(query_text)

    # Print the response to verify the output
    print(f"Response to query '{query_text}': {response}")


if __name__ == "__main__":
    main()
