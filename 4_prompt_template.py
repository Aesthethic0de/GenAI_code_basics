from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import os


load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


model = AzureChatOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], azure_deployment=os.environ["AZURE_DEPLYOMENT_NAME"],
                             api_key=os.environ["AZURE_OPENAI_KEY"], api_version=os.environ["AZURE_API_VERSION"])


#prompt_1
template = """tell me a joke about {topic}"""
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"topic" : "dogs"})
print(prompt)

#prompt 2
template_multipart = """you are a helpful assistant.
Human: tell me a {adjective} story about a {animal}"""
prompt_multi = ChatPromptTemplate.from_template(template_multipart)
prompt = prompt_multi.invoke({"adjective":"funny", "animal" : "panda"})
print(prompt)

#prompt3
messages = [
    ("system", "you are a comedian who tells a joke about {topic}"),
    ("human", "tell me a {joke_count} jokes")
]
prompt_list = ChatPromptTemplate.from_messages(messages)
prompt = prompt_list.invoke({"topic" : "lawyers", "joke_count" : 4})
print(prompt)


#prompt4
messages = [
    ("system", "you are a comedian who tells a joke about {topic}"),
    HumanMessage(content="tell me 3 jokes")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic" : "lawyers"})
print(prompt)
