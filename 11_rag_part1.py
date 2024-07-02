import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join("books", "test.txt")
persistent_directory = os.path.join(current_dir,"db", "chroma_db")

#PART - 1 Creating Chunks From Data

if not os.path.exists(persistent_directory):
    print("persistent directory does not exist. initializng vector store")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. please check the path")
    
    loader = TextLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents=documents)



    print(f"---Document Chunks information---")
    print(f"\n Number of document chunk : {len(docs)}")
    print(f"\n Sample chunk : {docs[0].page_content}")
    embeddings = AzureOpenAIEmbeddings(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_KEY"],
                                       openai_api_type="azure", azure_deployment="Embedding", show_progress_bar=True)
    
    print("--Creating Vector Store--")
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persistent_directory)
    print("---Fininshed creating vector store---")

else:
    print("--- vector store already exists ---")

