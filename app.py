import chainlit as cl
from agent.agent_core import create_agent
from dotenv import load_dotenv

load_dotenv()


@cl.on_chat_start
async def start():
    agent = create_agent()
    cl.user_session.set("agent", agent)
    await cl.Message(
        content="👋 Hi! I'm your Sales Enablement Assistant.\n\n"
                "Ask me about **products, demos, competitors, personas, pitch strategies**, "
                "or anything you need for your next customer call!"
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    agent = cl.user_session.get("agent")

    response = await cl.make_async(agent.invoke)({"input": message.content})

    await cl.Message(content=response["output"]).send()