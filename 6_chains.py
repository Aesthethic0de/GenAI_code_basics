from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import os
from langchain.schema.output_parser import StrOutputParser


load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


model = AzureChatOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], azure_deployment=os.environ["AZURE_DEPLYOMENT_NAME"],
                             api_key=os.environ["AZURE_OPENAI_KEY"], api_version=os.environ["AZURE_API_VERSION"])

prompt_template = ChatPromptTemplate.from_messages([

    ("system", "you are a comedian who tells joke about {topic}"),
    ("human", "tell me {joke_count} jokes")
])

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"topic" : "animal", "joke_count" : 2})

print(result)