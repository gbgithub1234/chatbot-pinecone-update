import os
import sys

from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from langchain.document_loaders import DirectoryLoader, TextLoader,UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredExcelLoader


import pinecone
from consts import INDEX_NAME

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


#------
#lang-chain tutorials



#------



#--------------------------------------------------------------------






def ingest_docs() -> None:
    # loader = ReadTheDocsLoader(
    #     path="langchain-docs/langchain.readthedocs.io/en/latest",
    #     encoding="utf8",
    #     features="html.parser",
    # )

    #----------------------
    #Load all of the PDF docs in the 'docs' folder

    loader = DirectoryLoader('docs', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) }documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    # for doc in documents:
    #     old_path = doc.metadata["source"]
    #     new_url = old_path.replace("langchain-docs", "https:/")
    #     doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****** Added to Pinecone vectorstore vectors")



    #----------------------
    #Load the Word docs in the 'docs' folder

    loader = DirectoryLoader('docs', glob="./*.docx", loader_cls=Docx2txtLoader)
    documents = loader.load()


    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) }documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    # for doc in documents:
    #     old_path = doc.metadata["source"]
    #     new_url = old_path.replace("langchain-docs", "https:/")
    #     doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****** Added to Pinecone vectorstore vectors")



if __name__ == "__main__":
    ingest_docs()
