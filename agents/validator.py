from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import json
import re

class ValidationInput(BaseModel):
    input_str: str = Field(..., description="JSON-encoded list of retrieved documents")

class ValidateSourcesTool(BaseTool):
    name: str = "validate_sources"
    description: str = "Evaluates clarity, relevance, and credibility of retrieved documents"
    args_schema: Type[BaseModel] = ValidationInput
    structured_output: bool = False

    def __init__(self, structured_output=False):
        super().__init__()
        self.structured_output = structured_output

    def _run(self, input_str: str) -> str:
        try:
            parsed = json.loads(input_str)
            documents = [
                Document(page_content=doc["page_content"], metadata=doc.get("metadata", {}))
                for doc in parsed
            ]
        except Exception as e:
            fallback = {"filtered": [], "commentary": f"❌ Failed to parse input: {e}"}
            return json.dumps(fallback) if self.structured_output else fallback["commentary"]

        llm = ChatOpenAI(model="gpt-4-turbo")

        text_blocks = "\n\n".join([
            f"Doc {i+1} (page {doc.metadata.get('page')}): {doc.page_content}"
            for i, doc in enumerate(documents)
        ])

        if self.structured_output:
            prompt = f"""

You are a research assistant.

Evaluate the following 5 documents for:
- clarity
- relevance to the research question
- credibility of content

Return ONLY a valid JSON object of the form:
{{
  "filtered": [
    {{
      "page_content": "...",
      "metadata": {{ "doc": "Doc 1", "page": 5 }}
    }},
    ...
  ],
  "commentary": "..."
}}

Do not include any markdown, code blocks, or extra commentary. Output only a JSON object.

{text_blocks}
"""



            

            def try_parsing_response(temp=0.7):
                res = llm.invoke(prompt, temperature=temp).content
                # Clean known issues: code blocks, preambles
                res_cleaned = re.sub(r"^```json|```$", "", res.strip())
                try:
                    json.loads(res_cleaned)
                    return res_cleaned
                except:
                    return None

            # Try parsing twice: normal + strict
            result = try_parsing_response(temp=0.7) or try_parsing_response(temp=0.0)
            if result:
                return result

            fallback = {
                "filtered": parsed,
                "commentary": "⚠️ Could not parse model output as JSON. Returning all documents."
            }
            return json.dumps(fallback)

        else:
            # ReAct-compatible (text only)
            prompt = f"""
You are a research assistant. Evaluate the following sources for:
- clarity
- relevance to the research question
- reliability of content

{text_blocks}

Write a concise review and flag weak or unclear sections.
            """
            return llm.invoke(prompt).content

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported yet.")

