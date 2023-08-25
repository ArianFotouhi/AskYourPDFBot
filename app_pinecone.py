from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
import os
import pinecone


os.environ["PINECONE_API_KEY"] = ""
os.environ["PINECONE_ENV"] = ""
os.environ["OPENAI_API_KEY"] = ""

loader = PyPDFLoader('Document.pdf')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()



# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  
    environment=os.getenv("PINECONE_ENV"),  
)

index_name = "arian-index"

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
      name=index_name,
      metric='cosine',
      dimension=768  
)

docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

query = "When did edgar alan poe travel to spain?"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)
