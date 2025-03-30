# agents/analyst.py

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import json

class AnalystInput(BaseModel):
    input_str: str = Field(..., description="JSON-encoded list of validated documents or a plain string")

class SynthesizeAnswerTool(BaseTool):
    name: str = "synthesize_answer"
    description: str = "Synthesizes a concise answer from validated research documents"
    args_schema: Type[BaseModel] = AnalystInput
    structured_output: bool = False

    def __init__(self, structured_output=False):
        super().__init__()
        self.structured_output = structured_output
    

    def _run(self, input_str: str) -> str:
        llm = ChatOpenAI(model="gpt-4-turbo")

        try:
            parsed = json.loads(input_str)
            print(f"Parsed input type: {type(parsed)}")

            if isinstance(parsed, dict) and "filtered" in parsed:
                docs = parsed["filtered"]
            elif isinstance(parsed, list):
                docs = parsed
            else:
                raise ValueError("Parsed input has unexpected structure.")

            documents = []
            for doc in docs:
                if isinstance(doc, Document):
                    documents.append(doc)
                elif isinstance(doc, dict) and "page_content" in doc:
                    documents.append(Document(
                        page_content=doc["page_content"],
                        metadata=doc.get("metadata", {})
                    ))
                else:
                    print(f"⚠️ Unexpected doc format: {doc}")
                    documents.append(Document(page_content=str(doc)))

        except Exception as e:
            print(f" Failed to parse input: {e}")
            print(" Fallback: treating input as single raw string document")
            documents = [Document(page_content=input_str)]

        try:
            combined_text = "\n\n".join(doc.page_content for doc in documents)
        except Exception as e:
            print(f" Failed to combine page content: {e}")
            print(f" Raw documents: {documents}")
            combined_text = str(documents)

        prompt = f"""
You are a research assistant helping synthesize insights from provided documents.

Based on the content below, write a concise, structured answer to the user's research question. 
Your output should:
- Address the key themes and findings relevant to the question
- Be clear, informative, and suitable for professional use
- Include supporting context where appropriate

Documents:
{combined_text}
        """

        return llm.invoke(prompt).content

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported yet.")
