from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel,set_tracing_disabled
import asyncio
import os
from dotenv import load_dotenv

set_tracing_disabled(disabled=True)
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
print("Using Gemini API Key:", gemini_api_key)



#Reference: https://ai.google.dev/gemini-api/docs/openai
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model=OpenAIChatCompletionsModel("gemini-2.5-flash",client)

spanish_agent = Agent(
    name="Spanish agent",
    instructions="Translate the user's message to Spanish.",
    model=model  # Pass the model here
)

french_agent = Agent(
    name="French agent",
    instructions="Translate the user's message to French.",
    model=model  # Pass the model here
)


orchestrator = Agent(
    name="Translator Orchestrator",
    instructions=(
        "You are a translation helper. If the user asks for Spanish, "
        "call translate_to_spanish. If French, call translate_to_french. "
        "Otherwise, ask which language they want."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish."
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French."
        ),
    ],
    model=model  # Pass the model here
)




async def main():
    result = await Runner.run(orchestrator, "Please say 'Good morning' in Spanish.")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())