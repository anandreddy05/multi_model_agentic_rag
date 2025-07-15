from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from dotenv import load_dotenv
from rag.loader import load_docs
import os


load_dotenv(override=True)

docs = load_docs("./data")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
split_docs = text_splitter.split_documents(docs)

persist_directory = os.path.join(os.getcwd(),"chroma_db")
os.makedirs(persist_directory,exist_ok=True)
collection_name = "Multi_Agentic_RAG"
try:
    vector_store = Chroma.from_documents(
        documents=split_docs,
        collection_name= collection_name,
        persist_directory=persist_directory,
        embedding=embeddings
    )
except:
    print("ChromaDB not Initialised")
    raise


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5}
)
