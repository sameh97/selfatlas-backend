from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

os.environ["GOOGLE_API_KEY"] = api_key

# Use "google_genai:" prefix instead of "google_vertexai:"
gemini_2_5_flash = init_chat_model("google_genai:gemini-2.5-flash", temperature=0)

response = gemini_2_5_flash.invoke("what's your name")
print(response.content)