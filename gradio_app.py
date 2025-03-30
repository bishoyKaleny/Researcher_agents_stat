import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from agents.retriver import load_retriever
from agents.validator import ValidateSourcesTool
from agents.analyst import SynthesizeAnswerTool
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import json

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# === Shared LLM + tools ===
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=openai_key, streaming=True)
tools = [load_retriever(), ValidateSourcesTool(), SynthesizeAnswerTool()]

# === ReAct Agent ===
react_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
react_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    memory=react_memory,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# === LangGraph Workflow ===
class GraphState(dict): pass

def build_graph():
    retriever = load_retriever()
    validator = ValidateSourcesTool(structured_output=True)
    synthesizer = SynthesizeAnswerTool(structured_output=True)

    def retrieve_node(state: GraphState) -> GraphState:
        result = retriever.run(state["question"])
        state["retrieved"] = result
        return state

    def validate_node(state: GraphState) -> GraphState:
        result = validator.run(state["retrieved"])
        state["validated"] = result
        return state

    def synthesize_node(state: GraphState) -> GraphState:
        result = synthesizer.run(state["validated"])
        state["answer"] = result
        return state

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("validate", validate_node)
    graph.add_node("synthesize", synthesize_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "validate")
    graph.add_edge("validate", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()

workflow_app = build_graph()

# === Gradio Chat Function with Chain-of-Thought ===
def chat_research(question, mode, chat_history):
    if not isinstance(chat_history, list):
        chat_history = []

    thoughts = []

    if mode == "ReAct":
        output = ""
        for chunk in react_agent.stream({"input": question}):
            if "intermediate_steps" in chunk:
                steps = chunk["intermediate_steps"]
                for action, obs in steps:
                    thoughts.append(
                        f"🔍 **Action**: {action.tool} → _{action.tool_input}_\n"
                        f"📝 **Observation**: {obs}"
                    )
            if "output" in chunk:
                output += chunk["output"]
                chat_history.append((question, output))
                return chat_history, "\n---\n".join(thoughts)

    else:  # Workflow mode
        state = workflow_app.invoke({"question": question})

        thoughts.append(" **Retrieved Documents**:")
        thoughts.append(f"```json\n{state.get('retrieved', 'None')}\n```")

        thoughts.append("**Validated Sources**:")
        thoughts.append(f"```json\n{state.get('validated', 'None')}\n```")

        answer = state.get("answer", " No answer generated.")
        chat_history.append((question, answer))
        return chat_history, "\n---\n".join(thoughts)

# === Gradio Interface ===
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# Research Assistant: ReAct & Workflow Modes")

    mode_selector = gr.Radio(["ReAct", "Workflow"], value="ReAct", label="Agent Mode")
    chatbot = gr.Chatbot()
    thoughts_box = gr.Markdown()
    question_input = gr.Textbox(placeholder="Ask a research question...", label="Your Question")
    submit_btn = gr.Button("Send")

    def user_asks(q, m, history):
        return chat_research(q, m, history)

    submit_btn.click(user_asks, [question_input, mode_selector, chatbot], [chatbot, thoughts_box])

if __name__ == "__main__":
    demo.queue().launch()
