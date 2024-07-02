import os
from dotenv import load_dotenv
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter
)

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings


load_dotenv()

embeddings = AzureOpenAIEmbeddings(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_KEY"],
                                       openai_api_type="azure", azure_deployment="Embedding", show_progress_bar=True)


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "test.txt")
db_dir = os.path.join(current_dir, "db")

loader = TextLoader(file_path=file_path)
documents = loader.load()

def create_vector_store(docs,store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persistent_directory)
        print(f"Finished creating vector store {store_name}")
    else:
        print(f"vector store {store_name} already exists")



# 1 - character based splitter
# splites text into chunks based on specified number of characters
# useful for consistent chunks sizes regardless of content structure
print("---Using Character-based Splitting---")
char_slpitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_slpitter.split_documents(documents)
create_vector_store(docs=char_docs, store_name="chorma_db_char")

#2 - sentence based splitting
# splites text into chunks based on sentences, ensuring chunks end at sentence boundaries.
# ideal for maintaining semantic coherence within chunks
print("---Sentence Based Splitting---")
sen_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sen_splitter.split_documents(documents)
create_vector_store(docs= sent_docs, store_name="chroma_db_sent")

#3 Token Based Splitting
# splits text ito chunks based on tokens (words, or subwords) , using tokenizers like GPT-2
# useful for transformers model with strict token limits
print("---Token Based Slitting---")
tok_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
token_docs = tok_splitter.split_documents(documents=documents)
create_vector_store(docs=token_docs, store_name="chroma_db_token")

#4 Recursive Character-based splitting
# attempt to split text at natural boundaries (sentence, pragraph) within character limits.
# balances between maintaining
print(f"---Using Recursive Character-Based Splitting---")
rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents=documents)
create_vector_store(rec_char_docs, "chroma_db_rec_char")

#5 custom Text splitting
class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split("\n\n")

custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents=documents)
create_vector_store(custom_docs, "chroma_db_customs")

def query_vector_store(store_name, query):
    persistent_dir = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_dir):
        print("\n ---Querying Vector Store--- \n")
        db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k" : 1, "score_threshold": 0.1},
        )
        print(f"\n\n --- Relevant Documents ---")
        relevant_docs = retriever.invoke(query)
        for i, doc in enumerate(relevant_docs,1):
            print(f"Document {i} : \n {doc.page_content}")
            if doc.metadata:
                print(f"\n Souce : {doc.metadata.get('source', 'Unknown')} \n")
    else:
        print(f"Vector Store {store_name} does not exists")


# #1 character based query
# query_vector_store(store_name="chorma_db_char", query="who is ron")
# #2 sentence based query
# query_vector_store(store_name="")

                                  
            

