import gradio as gr
from chat import TickTickChatbot
from typing import Dict, Optional
import json
from langchain_core.messages import HumanMessage, AIMessage


class ChatManager:
    """Manages multiple chat contexts"""

    def __init__(self):
        self.chats: Dict[str, TickTickChatbot] = {}
        self.current_chat_id: Optional[str] = None

    def get_or_create_bot(self, chat_id: str) -> TickTickChatbot:
        """Get existing bot or create new one for chat_id"""
        if chat_id not in self.chats:
            self.chats[chat_id] = TickTickChatbot()
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
        # Generate a chat ID from history if not exists
        chat_id = str(
            abs(hash(json.dumps(history[:1])) if history else id(history)))

        # Get or create bot for this chat
        bot = chat_manager.get_or_create_bot(chat_id)

        # If switching chats, rebuild bot's memory from history
        if chat_manager.current_chat_id != chat_id:
            bot.reset_memory()
            # Rebuild memory from history
            for h in history:
                if h["role"] == "user":
                    bot.memory.chat_memory.add_user_message(h["content"])
                elif h["role"] == "assistant":
                    bot.memory.chat_memory.add_ai_message(h["content"])
            chat_manager.switch_chat(chat_id)

        # Get response from bot
        response = await bot.chat(message)
        return response

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":

    demo = gr.ChatInterface(
        fn=chat_function,
        title="💬 TickTick Assistant",
        examples=["Show me my tasks", "Create a new task",
                    "What's on my schedule?"],
        submit_btn="Send",
        stop_btn=True,
        save_history=True,
        type="messages",
        fill_height=True,
        fill_width=True,
        autoscroll=True,
        show_progress="minimal",
        theme="default"
    )



    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
