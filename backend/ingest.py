from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

def get_embeddings_model() -> Embeddings:
    return GPT4AllEmbeddings()

# Add to vectorDB
vectorstore = Chroma.from_documents(
    persist_directory="./chroma_db",
    documents=all_splits,
    collection_name="rag-private",
    embedding=get_embeddings_model(),
)
retriever = vectorstore.as_retriever()
