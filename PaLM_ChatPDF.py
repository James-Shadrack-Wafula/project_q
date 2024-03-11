import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains import RetrievalQA

from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
import os, glob
from wBot2 import ask, init_f

init_f()

st.set_page_config(
    page_title="Q",  # Change the title displayed on the browser tab
    page_icon="🤖"  # Set the favicon (you can provide a URL or a unicode character)
)

# Sidebar contents
with st.sidebar:
    st.title("Jimmy's AI Assistant Q")
    # st.title("Ai Powered Revision Assistant")
    st.markdown(
        """
    ## About
    Hello meet Q, my AI Powerd Personal Assistant \n 
    Have a question? Ask Q. \n
    Contact me on:
    - [GitHub](https://github.com/James-Shadrack-Wafula/)
    - [WhatsApp](https://wa.me/254746727592/)
    - [Instagram](https://makersuite.google.com/app/home) 
 
    """
    )
    add_vertical_space(5)
    st.write("Developed by [Jimmy](http://james-shadrack-wafula.rf.gd/)")

load_dotenv()


def main():
    st.header("Chat with Q 💬")

    # files_path = "./SOURCE_DOCUMENTS/HUDUMAKENYADIGITALIZATIONPLAN-V1.pdf"
    # loaders = [UnstructuredPDFLoader(files_path)]

    # if "index" not in st.session:
    # index = VectorstoreIndexCreator(
    #     embedding=GooglePalmEmbeddings(),
    #     text_splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0),
    # ).from_loaders(loaders)

    # llm = GooglePalm(temperature=0.1)  # OpenAI()
    # chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=index.vectorstore.as_retriever(),
    #     # input_key="question",
    #     return_source_documents=True,
    # )

    # st.session.index = index
    answeres = []
    answere = ''
    # Accept user questions/query
    query = st.text_input("Ask questions:")
    # st.write(query)
    if query:
        response = ask(query)
        answeres.append({'query': [query,response]})
        answere = response
        # st.write('Write something')

        #===================================
        st.text(f"You: {query}")
        # Get and display the chatbot's response
        # bot_response = get_response(user_question)
        st.text(f"Q {answere}")

        #=====================================
        # with st.expander(query):
        # #  for answeres in response["source_documents"]:
        #     st.write(f"{response} ")
        #     for ans in answeres:
        #         st.write(ans["query"][0] + " " + ans["query"][1])


if __name__ == "__main__":
    main()
