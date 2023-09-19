from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
import os
from langchain.llms import HuggingFacePipeline

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ''


load_dotenv()

pdf_path = '/Document.pdf'

pdf_reader = PdfReader(pdf_path)
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

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-2-7b-hf",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64},
)


while True:
    user_question = input('Ask me: ')
    docs = knowledge_base.similarity_search(user_question)

    chain = load_qa_chain(llm,chain_type="stuff")
    response = chain.run(input_documents=docs,question=user_question)

    print("-----------------------------------------------Bot Reply------------------------------------")
    print(response)





