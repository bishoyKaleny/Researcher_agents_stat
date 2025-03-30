# main.py
from agents.retriver import load_retriever
from agents.analyst import SynthesizeAnswerTool
from agents.validator import ValidateSourcesTool

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain_core.documents import Document
import json
from dotenv import load_dotenv

import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
# === Load LLM ===
llm = ChatOpenAI(model="gpt-4-turbo",openai_api_key=openai_key)

# === Load tools ===
#retriever_tool = load_retriever()
tools = [load_retriever(), ValidateSourcesTool(), SynthesizeAnswerTool()]

# === Initialize agent ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# === Ask a question ===
question = "What are the current inflation trends in emerging Asian markets?"

# === Run the full workflow ===
print("\n🤖 Asking question:", question)
result = agent.invoke({"input": question})

# === Output final answer ===
print("\n✅ Final Synthesized Answer:\n")
print(result)
