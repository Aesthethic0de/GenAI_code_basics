from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import os
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel, RunnableBranch


load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


model = AzureChatOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], azure_deployment=os.environ["AZURE_DEPLYOMENT_NAME"],
                             api_key=os.environ["AZURE_OPENAI_KEY"], api_version=os.environ["AZURE_API_VERSION"])


positive_feedback = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant."),
    ("human", "generate a thank you note for this positive feedback: {feedback}")
])

negative_feedback = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant."),
    ("human", "generate a response addressing this negative feedback: {feedback}")
])


neutral_feedback = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant."),
    ("human", "generate a request for more details for this neutral feedback: {feedback}")
])


escalate_feedback = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant."),
    ("human", "generate a message to escalate this feedback to a human agent: {feedback}")
])

classification_template = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant"),
    ("human", "classify the sentiment of this feedback as positive, negative, neutral or escalate: {feedback}")
])

branches = RunnableBranch(
    (lambda x: "positive" in x,
     positive_feedback | model | StrOutputParser()),
    (lambda x: "negative" in x,
      negative_feedback | model | StrOutputParser()),
    (lambda x: "neutral" in x,
       neutral_feedback | model | StrOutputParser()),
    escalate_feedback | model | StrOutputParser()
)



classification_chain = classification_template | model | StrOutputParser()


chain = classification_chain | branches


result = chain.invoke({"feedback" : "No screen protector included in the package as shown in the name of the product. And the case they are advertising , well valve is the one providing the case which is free anyway so why is ninja taking credit ? Would appreciate some feedback on this. Thanx"})
print(result)