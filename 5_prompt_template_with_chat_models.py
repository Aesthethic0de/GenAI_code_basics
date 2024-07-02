from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import os


load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


model = AzureChatOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], azure_deployment=os.environ["AZURE_DEPLYOMENT_NAME"],
                             api_key=os.environ["AZURE_OPENAI_KEY"], api_version=os.environ["AZURE_API_VERSION"])

#prompt_1 simple template
template = "tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)
prompt =  prompt_template.invoke({"topic" : "engineers"})
result = model.invoke(prompt)
print(result.content)

#prompt_2 multiple place holders
template_multiparts = """you are a helpful assistant.
Human : tell me a {adjective} short story about a {animal}
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiparts)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal" : "dog"})
print(prompt)
result = model.invoke(prompt)
print(result.content)

#prompt3 prompt with system and human message using tuple

messages = [
    ("system", "you are a comedian who tells jokes about {topic}"),
    ("human" ,"tell me a {joke_count} jokes")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic" : "lawyer", "joke_count" : 5})
print(prompt)
result = model.invoke(prompt)
print(result.content)