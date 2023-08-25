from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ''
os.environ["OPENAI_API_KEY"] = ""

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask Your PDF")

    pdf = st.file_uploader("Please upload your pdf",type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # spilit ito chuncks
        text_splitter = CharacterTextSplitter(
            separator="\n", #new line is used to split the text
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embedding
        embeddings = HuggingFaceEmbeddings()

        knowledge_base = FAISS.from_texts(chunks,embeddings)

        user_question = st.text_input("Ask your questions about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
        
            #llm = HuggingFaceHub(repo_id= "google/flan-t5-large", model_kwargs={"temperature":0.5, "max_length":64})
            
            llm = ChatOpenAI( openai_api_key= os.getenv("OPENAI_API_KEY"), temperature=0, model_name="gpt-3.5-turbo")

            chain = load_qa_chain(llm,chain_type="stuff")
            response = chain.run(input_documents=docs,question=user_question)

            st.write(response)



        # st.write(chunks)

if __name__ == '__main__':
    main()
