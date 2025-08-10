import asyncio
from openai import AsyncOpenAI
from agents import Agent, set_default_openai_api, Runner,set_tracing_disabled,set_default_openai_client

import os
from dotenv import load_dotenv

set_tracing_disabled(disabled=True)
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
print("Using Gemini API Key:", gemini_api_key)

set_default_openai_api("chat_completions")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(external_client)

agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant", model="gemini-2.0-flash")

result = Runner.run_sync(agent, "Hello")

print(result.final_output)