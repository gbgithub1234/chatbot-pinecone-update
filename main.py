from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message


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


    # st.markdown("- created by Glen Brauer, Business Analyst in AAE")
    # st.markdown("- demonstrates the ability to leverage ChatGPT and vector storage to access documents")
    # st.markdown("- sample question: 'How can I create a marketing effort?'")
    #
    # url = "https://drive.google.com/drive/u/0/folders/1gTD-OiqH5Bg3-ZqVuur9q8h-AGIzOlB7"
    #
    # st.write("- documents which have been ingested are located [here](%s)" % url)

#------------------------------------------
st.header("SFU Document Chatbot 1.1 (beta)")


if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

#------------------------------------------

# st.help(st.form)


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string



with st.form(key='myform', clear_on_submit=True):
    prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
    submit_button = st.form_submit_button("Submit")
    



if submit_button:
    with st.spinner("Generating response.."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )




        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )


        #-----------------
        message(prompt, is_user=True)
        message(formatted_response)
        #-----------------


        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))



        # clear the inputbox
        # st.session_state["Prompt"] = ""
        # prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")



# if st.session_state["chat_answers_history"]:
#     for generated_response, user_query in zip(
#             st.session_state["chat_answers_history"],
#             st.session_state["user_prompt_history"],
#     ):
#         message(user_query, is_user=True)
#         message(generated_response)
