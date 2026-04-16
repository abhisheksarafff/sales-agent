import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agent.tools.kb_search import search_internal_kb
from agent.tools.web_search import search_web

load_dotenv()

SYSTEM_PROMPT = """You are a Sales Enablement Assistant for the field-facing team.
You help with: product knowledge, demo guidance, pitch strategies, competitor analysis, use cases, and buyer personas.

RULES:
1. Be concise and actionable — field teams need quick answers
2. Always cite your source when providing information
3. Use ONLY the context provided below - do not make up information"""


class SalesAgent:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
        self.chat_history = []

    def invoke(self, inputs: dict) -> dict:
        user_query = inputs.get("input", "")

        # Step 1: Search internal KB first
        print(f"\n🔍 Searching KB for: {user_query}")
        kb_results = search_internal_kb.invoke(user_query)
        print(f"📚 KB Results: {kb_results[:200]}...")

        # Step 2: Only search web if KB truly has nothing
        web_results = ""
        if "LOW_CONFIDENCE" in kb_results or "No relevant information found" in kb_results:
            print("🌐 KB empty, searching web...")
            web_results = search_web.invoke(user_query)
        else:
            print("✅ Using KB results only")

        # Step 3: Build context for LLM
        context = f"INTERNAL KNOWLEDGE BASE RESULTS:\n{kb_results}"
        if web_results:
            context += f"\n\nWEB SEARCH RESULTS (use only if KB has no answer):\n{web_results}"

        # Step 4: Generate response
        messages = [
            ("system", SYSTEM_PROMPT),
            ("system", f"Answer based on this context ONLY:\n\n{context}"),
        ]

        for msg in self.chat_history[-6:]:
            messages.append(msg)

        messages.append(("human", user_query))

        response = self.llm.invoke(messages)

        self.chat_history.append(("human", user_query))
        self.chat_history.append(("assistant", response.content))

        return {"output": response.content}


def create_agent():
    return SalesAgent()