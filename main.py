# ------------------------------------------
# Temporary Diagnostic: Direct Vector Retrieval Test
# ------------------------------------------
from pinecone import Pinecone as PineconeClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from consts import INDEX_NAME

if st.button("Run test: academic probation"):
    with st.spinner("Running similarity search..."):
        try:
            pc = PineconeClient(api_key=st.secrets["PINECONE_API_KEY"])
            index = pc.Index(INDEX_NAME)

            embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

            docsearch = LangchainPinecone.from_existing_index(
                index_name=INDEX_NAME,
                embedding=embeddings
            )

            results = docsearch.similarity_search("academic probation", k=5)

            if results:
                st.subheader("Top matching documents:")
                for i, doc in enumerate(results):
                    st.markdown(f"**Result {i+1}:**")
                    st.write(doc.page_content)
            else:
                st.warning("No matching documents found.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
