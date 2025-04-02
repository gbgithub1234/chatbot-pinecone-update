import os
import streamlit as st
from typing import Any, List, Dict

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone

from consts import INDEX_NAME


pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"],
    environment=st.secrets["PINECONE_ENVIRONMENT_REGION"],
    # api_key=os.environ["PINECONE_API_KEY"],
    # environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm(query: str, chat_history: List[Dict[str, Any]]=[]) -> Any:
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))
