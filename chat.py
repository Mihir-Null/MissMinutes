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
        # Initialize LLM with custom base URL if provided
        base_url = os.getenv("OPENAI_API_BASE")
        self.llm = ChatOpenAI(
            temperature=0.1,
            base_url=base_url if base_url else "https://api.openai.com/v1",
            model="gpt-4o-mini",
            streaming=True
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

IMPORTANT: The <summary> tag content is your FINAL ANSWER to the user. Everything outside the summary tag is your internal thought process and won't be shown to the user.

Example of good execution:
User: "Check tasks in Project A and B"

<think>
- Need to check tasks in two projects
- Will use get_project_tasks for each
- Will compile results into summary
</think>

<tool>get_project_tasks</tool>
<tool_input>
{{
    "project_id": "A"
}}
</tool_input>

<think>
- Project A has 2 tasks
- Both are high priority
- Need to check Project B next
</think>

<tool>get_project_tasks</tool>
<tool_input>
{{
    "project_id": "B"
}}
</tool_input>

<think>
- Project B has no tasks
- All information gathered
- Ready to provide final answer to user
</think>

<summary>
Here's what I found in your projects:

Project A:
- 2 high-priority tasks:
  1. Website Update (Due: Tomorrow)
  2. Email Campaign (Due: Friday)

Project B:
- No tasks currently assigned

Recommendation: Focus on the Website Update task due tomorrow, and consider assigning some tasks to Project B to keep it active.
</summary>

Remember:
- ALWAYS follow this structured process
- Everything outside <summary> tags is your internal thought process
- Only content inside <summary> tags will be shown to the user
- Make your final answer clear and actionable
- Maintain context from chat history"""),
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