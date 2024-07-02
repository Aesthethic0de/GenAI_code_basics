from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os


load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


chat_model = AzureChatOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], azure_deployment=os.environ["AZURE_DEPLYOMENT_NAME"],
                             api_key=os.environ["AZURE_OPENAI_KEY"], api_version=os.environ["AZURE_API_VERSION"])


messages = [
    SystemMessage(content="solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?")
]

result = chat_model.invoke(messages)

print(result.content)


messages_1 = [
    SystemMessage(content="solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="what is 18 times 5?")
]

result_1 = chat_model.invoke(messages_1)

print(result_1.content)
