from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os


load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


chat_model = AzureChatOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], azure_deployment=os.environ["AZURE_DEPLYOMENT_NAME"],
                             api_key=os.environ["AZURE_OPENAI_KEY"], api_version=os.environ["AZURE_API_VERSION"])

while True:
    ask = input("you:")
    result = chat_model.invoke(ask)
    print(result.content)
