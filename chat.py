from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import json
import re
import tools
import os
from dotenv import load_dotenv

class SimpleAgentParser(BaseOutputParser):
    def parse(self, text: str):
        """Parse the output text into agent actions or final answer.
        Prioritizes finding tool calls, then summary, then falls back to raw text.
        """
        # First priority: Find tool calls
        tool_pattern = r"<tool>(.*?)</tool>.*?<tool_input>\s*({.*?})\s*</tool_input>"
        tool_matches = list(re.finditer(tool_pattern, text, re.DOTALL))
        
        if tool_matches:
            # Get the last tool call
            tool_match = tool_matches[-1]
            tool_name = tool_match.group(1).strip()
            
            try:
                # Try to parse the JSON input, with cleanup
                tool_input_str = re.sub(r'\s+', ' ', tool_match.group(2).strip())
                tool_input = json.loads(tool_input_str)
                
                # Extract thinking for context if available
                think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                thinking = f"\n<think>\n{think_match.group(1).strip()}\n</think>\n" if think_match else ""
                
                # Format the log with proper XML
                log = f"{thinking}<tool>{tool_name}</tool>\n<tool_input>\n{json.dumps(tool_input, indent=2)}\n</tool_input>"
                
                return AgentAction(tool_name, tool_input, log)
            except json.JSONDecodeError:
                # If JSON parsing fails, try the next tool call if available
                if len(tool_matches) > 1:
                    # Remove the failed tool call and try again
                    new_text = text[:tool_match.start()] + text[tool_match.end():]
                    return self.parse(new_text)
        
        # Second priority: Check for final summary
        summary_match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
        if summary_match:
            return AgentFinish({"output": summary_match.group(1).strip()}, text)
        
        # Final fallback: Return the raw text
        # This helps with error messages and non-XML formatted responses
        return AgentFinish({"output": text.strip()}, text)

class TickTickChatbot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize LLM with custom base URL if provided
        base_url = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.llm = ChatOpenAI(
            temperature=0.1,
            base_url=base_url if base_url else "https://api.openai.com/v1",
            model="gpt-4o-mini",
            streaming=True,
            api_key=api_key
        )
        
        # Configure memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Set up tools
        self.tools = [
            tools.add_task,
            tools.complete_task,
            tools.get_task,
            tools.update_task,
            tools.delete_task,
            tools.get_project_tasks,
            tools.create_project,
            tools.get_project,
            tools.get_all_projects,
            tools.update_project,
            tools.delete_project,
            tools.get_active_projects,
            tools.get_closed_projects,
            tools.get_inbox_tasks,
            tools.get_project_tasks_detailed_with_data
        ]

        # Create a more explicit prompt that uses XML-style tags
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful TickTick assistant that helps manage tasks and projects. 
You have access to the following tools:

{tools}

Follow this exact process for EVERY request:

1. THINK: First, outline your plan
<think>
- What is the user asking for?
- What steps are needed?
- What tools will you use?
</think>

2. ACTIONS: For each step, use the appropriate tool with this format:
<tool>tool_name</tool>
<tool_input>
{{
    "param1": "value1",
    "param2": "value2"
}}
</tool_input>

3. THINK: After each tool response, analyze the result
<think>
- What did you learn?
- Was it successful?
- What's the next step?
</think>

4. REPEAT: Continue steps 2-3 until all needed information is gathered

5. FINAL ANSWER: When you have all information, provide your response to the user in a summary tag.
This will be shown directly to the user, so make it clear and complete.
<summary>
- Results from all operations
- Overall status
- Recommendations or next steps
</summary>

CRITICAL RULES:
1. NEVER output raw text without tags
2. ALWAYS wrap your thoughts in <think> tags
3. ALWAYS wrap your final response in <summary> tags
4. If you need to respond to the user, even for simple greetings, use <summary> tags
5. There should be NO text outside of tags
6. The internal data like id only useful for you for query but not useful to the user, do not include it in the summary. 

WRONG OUTPUTS:
1. "Let me check that for you"  (Wrong: raw text)
2. "<think>Checking tasks</think>
   I found 3 tasks"  (Wrong: mixed tagged and raw text)
3. "Hi there!"  (Wrong: greeting without tags)

CORRECT OUTPUTS:
1. "<think>Need to check tasks</think>"
2. "<think>Found 3 tasks, preparing response</think>
   <summary>I found 3 tasks in your inbox</summary>"
3. "<summary>Hi there! How can I help you today?</summary>"

Remember: EVERY single output must be wrapped in either <think>, <tool>, or <summary> tags. No exceptions.
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create the chain
        self.agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: self.memory.load_memory_variables({})["chat_history"],
                "agent_scratchpad": lambda x: self._format_scratchpad(x["intermediate_steps"]),
                "tools": lambda x: "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            }
            | self.prompt
            | self.llm
            | SimpleAgentParser()
        )

        # Create the executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def _format_scratchpad(self, intermediate_steps):
        """Format intermediate steps into a list of messages."""
        if not intermediate_steps:
            return []
        
        messages = []
        for action, observation in intermediate_steps:
            messages.append(
                AIMessage(content=action.log)
            )
            messages.append(
                HumanMessage(content=f"Tool output: {observation}")
            )
        
        return messages

    async def chat(self, message: str) -> str:
        """Process a user message and return the complete response"""
        try:
            # Execute the agent
            response = await self.agent_executor.ainvoke({"input": message})
            return response["output"]
            
        except Exception as e:
            print(f"\nError in chat: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return f"Error processing your request: {str(e)}"

    async def chat_stream(self, message: str):
        """Process a user message and stream the response chunks"""
        try:
            # Execute the agent with streaming
            async for chunk in self.agent_executor.astream(
                {"input": message},
            ):
                if "output" in chunk:
                    yield chunk["output"]
                
        except Exception as e:
            print(f"\nError in chat: {str(e)}")
            print(f"Error type: {type(e)}")
            yield f"Error processing your request: {str(e)}"

    def reset_memory(self):
        """Reset the conversation memory"""
        self.memory.clear()
        return "Memory cleared successfully"

    def save_memory(self, file_path: str):
        """Save the conversation memory to a file"""
        try:
            memory_data = self.memory.load_memory_variables({})
            with open(file_path, 'w') as f:
                json.dump(memory_data, f, default=lambda x: x.dict() if hasattr(x, 'dict') else str(x))
            return f"Memory saved to {file_path}"
        except Exception as e:
            return f"Error saving memory: {str(e)}"

    def load_memory(self, file_path: str):
        """Load conversation memory from a file"""
        try:
            with open(file_path, 'r') as f:
                memory_data = json.load(f)
            
            # Convert loaded data back to messages
            messages = []
            for msg in memory_data.get("chat_history", []):
                if msg.get("type") == "human":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("type") == "ai":
                    messages.append(AIMessage(content=msg["content"]))
            
            # Reset current memory and add loaded messages
            self.memory.clear()
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    self.memory.chat_memory.add_user_message(msg.content)
                elif isinstance(msg, AIMessage):
                    self.memory.chat_memory.add_ai_message(msg.content)
            
            return f"Memory loaded from {file_path}"
        except Exception as e:
            return f"Error loading memory: {str(e)}"

    async def chat_with_metadata(self, message: str):
        """Process a user message and yield responses with metadata in real-time"""
        try:
            async for chunk in self.agent_executor.astream({"input": message}):
                try:
                    # 1. Handle thinking patterns
                    if "output" in chunk:
                        output = chunk["output"]
                        if isinstance(output, str):  # Ensure output is string
                            think_match = re.search(r"<think>(.*?)</think>", output, re.DOTALL)
                            if think_match:
                                yield {
                                    "role": "assistant",
                                    "content": think_match.group(1).strip(),
                                    "metadata": {"title": "🧠 Thinking"}
                                }
                                if all(tag not in output for tag in ["<tool>", "<summary>"]):
                                    continue  # Skip if only thinking

                    # 2. Handle tool actions
                    if "actions" in chunk and chunk["actions"]:
                        for action in chunk["actions"]:
                            if not hasattr(action, 'log') or not hasattr(action, 'tool'):
                                continue  # Skip invalid actions
                                
                            # Extract thinking from action
                            if action.log:
                                think_match = re.search(r"<think>(.*?)</think>", action.log, re.DOTALL)
                                if think_match:
                                    yield {
                                        "role": "assistant",
                                        "content": think_match.group(1).strip(),
                                        "metadata": {"title": "🧠 Thinking"}
                                    }

                            # Format tool usage
                            tool_input = getattr(action, 'tool_input', {})
                            if isinstance(tool_input, (dict, str)):
                                yield {
                                    "role": "assistant",
                                    "content": f"Using {action.tool}\nInput: {json.dumps(tool_input, indent=2)}",
                                    "metadata": {"title": "🔧 Tool"}
                                }

                    # 3. Handle tool results
                    if "steps" in chunk and chunk["steps"]:
                        for step in chunk["steps"]:
                            if hasattr(step, 'observation') and step.observation:
                                yield {
                                    "role": "assistant",
                                    "content": str(step.observation),
                                    "metadata": {"title": "📝 Result"}
                                }

                    # 4. Handle final response
                    if "output" in chunk and "messages" in chunk and chunk["messages"]:
                        message_content = chunk["messages"][0].content if chunk["messages"] else ""
                        if message_content:
                            summary_match = re.search(r"<summary>(.*?)</summary>", message_content, re.DOTALL)
                            if summary_match:
                                yield {
                                    "role": "assistant",
                                    "content": summary_match.group(1).strip(),
                                    "metadata": {"title": "💬 Response"}
                                }
                            elif chunk["output"] and not any(tag in chunk["output"] for tag in ["<think>", "<tool>"]):
                                # Only yield raw output if it's not thinking or tool usage
                                yield {
                                    "role": "assistant",
                                    "content": str(chunk["output"]),
                                    "metadata": {"title": "💬 Response"}
                                }

                except Exception as chunk_error:
                    print(f"Error processing chunk: {chunk_error}")
                    continue  # Skip problematic chunk but continue processing
                
        except Exception as e:
            print(f"\nError in chat: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            yield {
                "role": "assistant",
                "content": f"Error processing your request: {str(e)}",
                "metadata": {"title": "❌ Error"}
            }