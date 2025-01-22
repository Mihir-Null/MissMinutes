from dotenv import load_dotenv, find_dotenv
import os

# Force reload of environment variables
load_dotenv(find_dotenv(), override=True)

# OpenAI-compatible API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # Your custom API endpoint 