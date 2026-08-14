from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split PDF documents into smaller chunks
    for embedding and retrieval.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    return chunks