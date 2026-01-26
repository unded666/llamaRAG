from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
# from langchain_classic.document_loaders import PyPDFLoader
# from langchain.s import PyPDFLoader

import os

PHB_FILE = './Docs/Phb.pdf'
PHB_DB_LOCATION = './chroma_phb_db'

def embed_file (file_path: str = PHB_FILE,
                db_location: str = PHB_DB_LOCATION):
    """Embed the PHB PDF into a Chroma vector store."""

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    embeddings = OllamaEmbeddings(model = 'mxbai-embed-large')

    add_documents = not os.path.exists(db_location)

    vector_store = Chroma(
        collection_name = 'phb',
        persist_directory=db_location,
        embedding_function=embeddings
    )

    if add_documents:
        ids = [str(i) for i in range(len(pages))]
        vector_store.add_documents(documents = pages,
                                   ids = ids)

    return vector_store.as_retriever(
        search_kwargs = {"k": 5}
    )