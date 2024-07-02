import os
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "db", "chromadb_with_metadata")

embeddings = AzureOpenAIEmbeddings(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_KEY"],
                                       openai_api_type="azure", azure_deployment="Embedding", show_progress_bar=True)

db = Chroma(persist_directory=persistent_dir,
            embedding_function=embeddings)

query = "who is ron?"

retriever = db.as_retriever(search_type="similarity_score_threshold",
                            search_kwargs={"k":5, "score_threshold" : 0.1})

relevant_docs = retriever.invoke(query)

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i} : \n {doc.page_content} \n")
    print(f"Source : {doc.metadata['source']} \n")

                                    