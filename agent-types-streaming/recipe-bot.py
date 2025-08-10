
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
import os
from dotenv import load_dotenv

set_tracing_disabled(disabled=True)
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
print("Using Gemini API Key:", gemini_api_key)


# Client Setup for Connecting to Gemini
external_client:AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

#Initialize model
model:OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

def main():
  # Create the Recipe Agent
  agent = Agent(
      name="RecipeBot",
      instructions=(
          """You are a helpful recipe assistant. A user will give you a few ingredients
          they have at home, and you will suggest one simple and quick recipe using only those items.
          Keep it short, step-by-step, and easy for beginners to cook."""
      ),
      model=model
  )

  print("\n🍳 What can I cook today?\n")
  ingredients = "eggs, tomatoes, onions, and bread"
  result:Runner = Runner.run_sync(agent, f"I have these at home: {ingredients}. What can I cook?")

  print(result.final_output)



main()