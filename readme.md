# TickTick Assistant

A chatbot interface for managing TickTick tasks and projects using natural language, built with LangChain and Dida365 package.

## Prerequisites

- Python 3.8+
- TickTick account
- OpenAI API key or compatible API endpoint
- TickTick developer account (for OAuth2 credentials)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Cyfine/Lang-TickTick.git
cd Lang-TickTick
```

2. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # Optional: custom endpoint
```

3. Configure TickTick credentials:
   - Log in to your TickTick account through the dida365-api library
   - The client will handle authentication automatically on first use

## Running the Application

1. Start the web interface:
```bash
python web.py
```

2. Open your browser and navigate to:
```
http://localhost:7860
```