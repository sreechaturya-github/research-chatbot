from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

# Wikipedia utility (plain Python call, no tool binding)
wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

def run_agent(query: str) -> ResearchResponse:
    wiki_result = wiki.run(query)

    prompt = (
        f"You are a knowledgeable research assistant. Answer the following query using your own knowledge "
        f"and the Wikipedia context provided. Do NOT say the information is not in the context — "
        f"use your general knowledge to fill in any gaps.\n\n"
        f"Query: {query}\n\n"
        f"Wikipedia Context (use as reference, not as a limit):\n{wiki_result}\n\n"
        f"If the query asks for a list (like top 5, best 10, etc.), always provide a proper numbered list in the summary.\n\n"
        f"Respond ONLY with a JSON object, no extra text, no markdown:\n"
        + parser.get_format_instructions()
    )

    response = llm.invoke(prompt)
    return parser.parse(response.content)
def suggest_questions(topic: str, summary: str) -> list[str]:
    prompt = (
        f"Based on this research summary about '{topic}':\n{summary}\n\n"
        "Generate exactly 3 short, interesting follow-up questions a curious person might ask next. "
        "Return only a JSON array of 3 strings, no extra text. Example: [\"Q1?\", \"Q2?\", \"Q3?\"]"
    )
    response = llm.invoke(prompt)
    import json, re
    match = re.search(r'\[.*?\]', response.content, re.DOTALL)
    return json.loads(match.group()) if match else []