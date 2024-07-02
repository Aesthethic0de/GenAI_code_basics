from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import os
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel


load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


model = AzureChatOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], azure_deployment=os.environ["AZURE_DEPLYOMENT_NAME"],
                             api_key=os.environ["AZURE_OPENAI_KEY"], api_version=os.environ["AZURE_API_VERSION"])

prompt_template = ChatPromptTemplate.from_messages([

    ("system", "you are a expert product reviewer"),
    ("human", "list the main features of the product {product_name}")
])

def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages([
        ("system", "you are an expert product reviewer"),
        ("human", "given these features: {features}, list the pros of these features")
    ])
    return pros_template.format_prompt(features=features)


def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages([
        ("system", "you are an expert product reviewer"),
        ("human", "given these features: {features}, list the cons of these features")
    ])
    return cons_template.format_prompt(features=features)

def combine_pros_and_cons(pros,cons):
    return f"Pros: {pros}\n\n Cons: {cons}"

pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chains = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)


chain = (prompt_template
         | model
         | StrOutputParser()
         | RunnableParallel(branches={"pros" : pros_branch_chain, "cons" : cons_branch_chains})
         | RunnableLambda(lambda x: combine_pros_and_cons(x["branches"]["pros"], x["branches"]["cons"])))



result = chain.invoke({"product_name" : "Macbook Pro"})

print(result)

