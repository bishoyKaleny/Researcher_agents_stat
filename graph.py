# main_langgraph.py

from agents.retriver import load_retriever
from agents.validator import ValidateSourcesTool
from agents.analyst import SynthesizeAnswerTool

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import json

# === Load environment variables ===
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# === Load LLM ===
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=openai_key)

# === Load tools ===
retriever_tool = load_retriever()
validator_tool = ValidateSourcesTool(structured_output=True)
synthesizer_tool = SynthesizeAnswerTool(structured_output=True)

# === Define LangGraph state ===
class GraphState(dict):
    pass

# === Define each node ===

def retrieve_node(state: GraphState) -> GraphState:
    question = state["question"]
    result = retriever_tool.run(question)
    state["retrieved"] = result
    print("\n📦 Retrieved Context:\n", result)
    return state

def validate_node(state: GraphState) -> GraphState:
    result = validator_tool.run(state["retrieved"])
    state["validated"] = result
    print("\n✅ Validation Output:\n", result)
    return state

def synthesize_node(state: GraphState) -> GraphState:
    result = synthesizer_tool.run(state["validated"])
    state["answer"] = result
    return state

# === Build the graph ===
graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("validate", validate_node)
graph.add_node("synthesize", synthesize_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "validate")
graph.add_edge("validate", "synthesize")
graph.add_edge("synthesize", END)

# === Compile the graph ===
app = graph.compile()

# === Ask a question ===
question = "What are the current inflation trends in emerging Asian markets?"

# === Run the graph ===
print("\n🤖 Asking question:", question)
final_state = app.invoke({"question": question})

# === Output final answer ===
print("\n✅ Final Synthesized Answer:\n")
print(final_state["answer"])
