from typing import Set
import streamlit as st
from streamlit_chat import message
from backend.core import run_llm
from consts import INDEX_NAME

#------------------------------------------
url = "https://drive.google.com/drive/u/0/folders/1gTD-OiqH5Bg3-ZqVuur9q8h-AGIzOlB7"
url2 = "https://www.pinecone.io/"

multiline_str1 = """

- created by Glen Brauer, Business Analyst in AAE (glenb@sfu.ca) \n

- PROBLEM: document-based information is located in many places taking time to find\n

- SOLUTION: provide a one-stop shopping resource for all document-based information\n

- leverages AI and [Pinecone vector storage](%s) """ % url2

multiline_str2 = """to access these [sample documents](%s)""" % url
multiline_str3 ="""\n - sample prompt: 'How can I create a marketing effort?' \n"""

with st.expander("Show/hide details"):
    st.write(multiline_str1 + multiline_str2 + multiline_str3)

#------------------------------------------
st.header("SFU Document Chatbot 1.2 (beta)")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

#------------------------------------------

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

#------------------------------------------
# Main chatbot UI
#------------------------------------------

with st.form(key='myform', clear_on_submit=True):
    prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    with st.spinner("Generating response..."):
        try:
            generated_response = run_llm(
                query=prompt, chat_history=st.session_state["chat_history"]
            )

            if isinstance(generated_response, dict):
                source_docs = generated_response.get("source_documents", [])
                sources = set(doc.metadata.get("source", "Unknown") for doc in source_docs)
                answer = generated_response.get("answer", "[No answer generated]")
            else:
                source_docs = []
                sources = set()
                answer = generated_response if generated_response else "[No answer generated]"

            formatted_response = f"{answer} \n\n {create_sources_string(sources)}"

            message(prompt, is_user=True)
            message(formatted_response)

            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["chat_history"].append((prompt, answer))

        except Exception as e:
            st.error(f"An error occurred during response generation: {e}")

#------------------------------------------
# Vector search test (now commented out)
#------------------------------------------

# with st.expander("Run Vector Search Test"):
#     if st.button("Run test: What is a marketing plan"):
#         with st.spinner("Running similarity search..."):
#             try:
#                 from pinecone import Pinecone as PineconeClient
#                 from langchain.embeddings.openai import OpenAIEmbeddings
#                 from langchain_community.vectorstores import Pinecone as LangchainPinecone
#
#                 pc = PineconeClient(api_key=st.secrets["PINECONE_API_KEY"])
#                 index = pc.Index(INDEX_NAME)
#
#                 embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
#
#                 docsearch = LangchainPinecone.from_existing_index(
#                     index_name=INDEX_NAME,
#                     embedding=embeddings
#                 )
#
#                 query = "What is a marketing plan"
#                 results = docsearch.similarity_search(query, k=5)
#
#                 if results:
#                     st.subheader("Top matching documents for:")
#                     st.markdown(f"**Query:** {query}")
#                     for i, doc in enumerate(results):
#                         st.markdown(f"**Result {i+1}:**")
#                         st.write(doc.page_content)
#                         if doc.metadata:
#                             st.caption(f"📄 Source: {doc.metadata.get('source', 'Unknown')}")
#                 else:
#                     st.warning("No matching documents found.")
#
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")
