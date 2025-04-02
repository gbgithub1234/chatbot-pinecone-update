# backend/core.py

import streamlit as st
from typing import Any, List, Dict

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Pinecone  # correct path for v3
from pinecone import Pinecone as PineconeClient  # ✅ new SDK v3

from consts import INDEX_NAME

# ✅ Initialize Pinecone v3 client (no environment or init needed)
pc = PineconeClient(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)  # This confirms access to the migrated serverless index


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    try:
        result = qa_chain({
            "question": query,
            "chat_history": chat_history
        })

        # Log what actually came back
        #st.write("Raw result from LangChain:", result) 
        #disabling this for now - GB

        return result or {"answer": "[LangChain returned no result]", "source_documents": []}

    except Exception as e:
        st.error(f"Error in LLM chain: {e}")
        return {"answer": "[Error occurred during LLM processing]", "source_documents": []}

