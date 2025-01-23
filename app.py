import gradio as gr
from chat import TickTickChatbot
from typing import Dict, Optional
import json
from langchain_core.messages import HumanMessage, AIMessage
import os
from pathlib import Path
import asyncio


class ChatManager:
    """Manages multiple chat contexts"""

    def __init__(self):
        self.chats: Dict[str, TickTickChatbot] = {}
        self.current_chat_id: Optional[str] = None
        # Add storage path
        self.storage_dir = Path("chat_history")
        self.storage_dir.mkdir(exist_ok=True)

    def _get_chat_file(self, chat_id: str) -> Path:
        """Get path for chat history file"""
        return self.storage_dir / f"chat_{chat_id}.json"

    async def save_chat_history(self, chat_id: str):
        """Save chat history to file"""
        bot = self.chats.get(chat_id)
        if bot:
            file_path = self._get_chat_file(chat_id)
            bot.save_memory(str(file_path))

    async def load_chat_history(self, chat_id: str) -> TickTickChatbot:
        """Load chat history from file if exists"""
        file_path = self._get_chat_file(chat_id)
        bot = self.get_or_create_bot(chat_id)
        if file_path.exists():
            bot.load_memory(str(file_path))
        return bot

    def get_or_create_bot(self, chat_id: str) -> TickTickChatbot:
        """Get existing bot or create new one for chat_id"""
        if chat_id not in self.chats:
            self.chats[chat_id] = TickTickChatbot()
            # Try to load existing history
            asyncio.create_task(self.load_chat_history(chat_id))
        return self.chats[chat_id]

    def switch_chat(self, chat_id: str):
        """Switch to a different chat context"""
        self.current_chat_id = chat_id
        return self.get_or_create_bot(chat_id)


# Global chat manager
chat_manager = ChatManager()


async def chat_function(message: str, history):
    """Core chat function that processes messages and maintains history"""
    try:
        chat_id = str(abs(hash(json.dumps(history[:1])) if history else id(history)))
        bot = await chat_manager.load_chat_history(chat_id)
        
        if chat_manager.current_chat_id != chat_id:
            bot.reset_memory()
            for h in history:
                if h["role"] == "user":
                    bot.memory.chat_memory.add_user_message(h["content"])
                elif h["role"] == "assistant":
                    bot.memory.chat_memory.add_ai_message(h["content"])
            chat_manager.switch_chat(chat_id)

        # Keep track of all messages in this response
        response_messages = []
        async for msg in bot.chat_with_metadata(message):
            response_messages.append({
                "role": "assistant",
                "content": msg["content"],
                "metadata": msg.get("metadata", {})
            })
            # Yield all messages so far as a list
            yield response_messages
        
        await chat_manager.save_chat_history(chat_id)

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield [{
            "role": "assistant", 
            "content": f"Error: {str(e)}", 
            "metadata": {"title": "❌ Error"}
        }]

if __name__ == "__main__":


    # Create ChatInterface with custom chatbot
    demo = gr.ChatInterface(
        fn=chat_function,
        # chatbot=chatbot,
        title="💬 TickTick Assistant",
        description="Your AI assistant for managing tasks and projects in TickTick",
        examples=["Show me my tasks", "Create a new task", "What's on my schedule?"],
        submit_btn="Send",
        stop_btn=True,
        save_history=True,
        type="messages",
        fill_height=True,
        fill_width=True,
        autoscroll=True,
        show_progress="minimal",
        theme="default",
    )

    demo.queue()
    demo.launch(server_name="localhost", server_port=7860, show_error=True)
