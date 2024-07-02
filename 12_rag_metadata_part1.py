import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir, "chromadb_with_metadata")

print(f" Books directory -- {books_dir}")
print(f"Chromadb metadata directory -- {persistent_dir}")

if not os.path.exists(persistent_dir):
    print("chromadb metadata doesnot exists")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"The directory {books_dir} does not exists!!")
    
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    documents = []
    for i in book_files:
        file_path = os.path.join(books_dir, i)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": i}
            documents.append(doc)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n---Document chunks information---")
    print(f"\n number of document chuks : {len(docs)}")

    embeddings = AzureOpenAIEmbeddings(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_KEY"],
                                       openai_api_type="azure", azure_deployment="Embedding", show_progress_bar=True)
    
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persistent_dir)
    print("Fininshed creating vector store")
else:
    print("vector store already exists")
    



