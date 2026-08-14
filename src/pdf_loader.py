from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    """
    Load PDF and return LangChain documents.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents