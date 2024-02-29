from document_processor import DocumentProcessor


def test_document_processor():
    print("Testing DocumentProcessor...")
    processor = DocumentProcessor()
    processor.load_documents()
    processor.build_agents()

    assert len(processor.documents) > 0, "No documents were loaded."
    assert len(processor.agents_dict) > 0, "Agents were not built correctly."
    print("DocumentProcessor loaded documents and built agents successfully.")


if __name__ == "__main__":
    test_document_processor()
