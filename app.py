import os
import chainlit as cl
from agent.agent_core import create_agent
from dotenv import load_dotenv

load_dotenv()


def ensure_vectorstore():
    if not os.path.exists("./vectorstore") or not os.listdir("./vectorstore"):
        print("Vectorstore not found, creating...")
        from ingestion.ingest import ingest_documents
        ingest_documents()


ensure_vectorstore()


@cl.on_chat_start
async def start():
    agent = create_agent()
    cl.user_session.set("agent", agent)
    await cl.Message(
        content="Hi! I'm your Sales Enablement Assistant.\n\n"
                "Ask me about products, demos, competitors, personas, or pitch strategies!"
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    response = await cl.make_async(agent.invoke)({"input": message.content})
    await cl.Message(content=response["output"]).send()