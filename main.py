from langgraph.graph import START, END, add_messages, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Sequence, Annotated, TypedDict, Optional, List, Any
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage, ToolMessage
from langchain.tools import tool
from dotenv import load_dotenv
from numpy import lib
from pydantic import BaseModel, Field
import os

from tools.calculator import calculator
from tools.python_code_executor import Python_Code_Executor
from tools.web_search import web_search
from tools.save import save_tool
from tools.retriever import retriever_tool
from tools.summarize import summarize_tool

load_dotenv(override=True)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    feedback: bool
    eval_feedback: bool
    retry_count: int
    clarified_query: Optional[str]
    llm_response: Optional[str]
    evaluator_response: Optional[str]
    tool_use: bool

class Evaluator(BaseModel):
    eval_feedback: bool = Field(description="Whether the response is correct or not")
    reason: str = Field(description="Reason for the feedback")

class Agent:
    def __init__(self):
        self.llm = None
        self.llm_with_tools = None
        self.evaluator = None
        self.evaluate_response = None
        self.tools = None
        self.tool_node = None
        self.graph = None
        self.memory = None
        
    def setup(self):
        # Define tools at class level
        self.tools = [retriever_tool, calculator, Python_Code_Executor, web_search, save_tool, summarize_tool]
        self.tool_node = ToolNode(self.tools)
        
        # Initialize LLMs
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
        self.evaluator = ChatOpenAI(model="gpt-4o-mini").with_structured_output(Evaluator)
        self.evaluator_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools=self.tools).with_structured_output(Evaluator)
        
        # Initialize graph and memory
        self.graph = StateGraph(State)
        self.memory = MemorySaver()
    
    def agent(self, state: State) -> State:
        print("🧠 Agent node executed. Proceeding to tool check or evaluation.")
        system_prompt = f"""
            You are a helpful and intelligent assistant equipped with a set of powerful tools. 
            Your goal is to respond accurately, efficiently, and transparently based on the user's query. 
            You are part of a feedback-driven system where your answers may be evaluated and compared, either by a user or by an automated evaluator.

            You have access to the following tools:

            1. `retriever_tool` – Retrieve answers from a local knowledge base of documents (e.g., manuals, reports, notes).
            2. `web_search` – Use this for current or real-time information (e.g., news, weather, stock prices).
            3. `calculator` – Use for arithmetic or numeric calculations (e.g., 22 / 7, 10 * 4.5).
            4. `Python_Code_Executor` – Use this for running simple Python code (e.g., loops, conditionals, expressions).
            ⚠️ Do NOT use or suggest unsafe modules like `os`, `open`, or `import`.
            5. `save_tool` – Use only if the user explicitly asks you to save a response to a file.
            6. `summarize_tool` – Use this to summarize long outputs when needed or upon request.

            ### System Behavior Guidelines:

            - If the query involves real-time data (e.g., current weather, news, events), always prefer using the `web_search` tool instead of relying on prior knowledge.
            - Use tools only when relevant to the query — avoid unnecessary calls.
            - Clearly indicate when a tool was used and summarize its contribution to your answer.
            - You may be evaluated and compared to another evaluator's response. Focus on accuracy and correctness.
            - This is the feedback you got from the evaluator: {state.get('evaluator_response', 'None')}
            - Your response will be stored and may be re-evaluated if the user is not satisfied.
            - If the question is not clear, ask the user to clarify if {state.get('retry_count', 0)} < 3.
            - If a user provides feedback or clarification, treat it as a new message and reprocess accordingly use {state.get('clarified_query', 'None')} as the query.
            - If saving is requested, send a clear and formatted output suitable for file storage.
            - Keep answers concise, but support with factual reasoning or evidence when needed.

            IMPORTANT: When you need to use a tool, you MUST call it. Don't just describe what the tool would do - actually use it!

            Respond intelligently and transparently. Prioritize correctness, efficiency, and user satisfaction.
            """
        
        if self.llm_with_tools is None:
            raise ValueError("Agent not initialized. Please call agent.setup() before using agent().")

        messages = state['messages'].copy()  

        # Add system prompt only if not already present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages.insert(0, SystemMessage(content=system_prompt))
        
        # Handle clarified query properly
        if state.get("clarified_query") and state.get("retry_count", 0) > 0:
            messages.append(HumanMessage(content=state["clarified_query"]))
        
        response = self.llm_with_tools.invoke(messages)
        
        # Track tool usage
        tool_use = hasattr(response, "tool_calls") and len(response.tool_calls) > 0
        tool_use = state.get("tool_use", False) or tool_use 
        return {
            **state,
            "messages": messages + [response],  
            "llm_response": response.content,
            "tool_use": tool_use,
        }
        
    def llm_router(self, state: State):
        """Route based on whether tools were called"""
        print("➡️ LLM Router is Being Used")
        last_message = state['messages'][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_used"
        else:
            return "evaluation"
    
    def format_conversation(self, state: State):
        """Format conversation history for evaluator"""
        conversation = "Conversation History:\n\n"
        for message in state['messages']:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                conversation += f"Assistant: {message.content}\n"
            elif isinstance(message, ToolMessage):
                conversation += f"Tool: {message.content}\n"
        print("🗂️ Formatted Conversation:\n" + conversation)
        return conversation

    
    def evaluate(self, state: State):
        print("🧪 Evaluator without tools running...")
        eval_prompt = SystemMessage(content=f"""
            You are an intelligent evaluator tasked with judging the quality and correctness of an AI assistant's response. 
            You do **not** have access to any tools and must evaluate purely based on the conversation history and the assistant's final response.

            Your job:
            - Verify whether the assistant's answer is factually accurate, logically sound, and aligned with the user's query.
            - Provide a structured judgment.

            Here is the full conversation history:
            {self.format_conversation(state)}

            Final assistant response:
            {state.get('llm_response', '')}

            ## Instructions:
            - Carefully read the assistant's answer.
            - Identify any factual or logical errors.
            - Do **not** attempt to fix or rewrite the answer.
            - Be unbiased in your assessment.

            ## Respond using this structured format:
            - eval_feedback: `True` if the answer is valid, `False` if incorrect.
            - reason: A brief justification for your decision (1-2 lines).
            """)

        evaluator_messages = [eval_prompt] + state["messages"]
        
        # Get structured evaluation
        result: Evaluator = self.evaluator.invoke(evaluator_messages)
        print(f"🧾 Evaluator Result: {result.eval_feedback}, Reason: {result.reason}")

        
        return {
            **state,
            "evaluator_response": result.reason,
            "eval_feedback": result.eval_feedback,
            "tool_use": state.get("tool_use", False)
        }
    
    def evaluate_with_tools(self, state: State):
        """Evaluate the assistant's response with tools"""
        print("🧪 Evaluator WITH tools running...")
        
        if self.evaluator_with_tools is None:
            raise ValueError("Agent not initialized. Please call agent.setup() before using agent().")
        
        eval_prompt = SystemMessage(content=f"""
            You are an intelligent evaluator responsible for validating the quality and correctness of an AI assistant's response to a user's query. 
            You have access to tools for verification and may use them as needed.

            Here is the full conversation history:
            {self.format_conversation(state)}

            Final assistant response:
            {state.get('llm_response', '')}

            ## Tools available:
            1. `retriever_tool` – To access internal documents.
            2. `web_search` – For verifying real-time or factual info.
            3. `calculator` – For math or numeric validations.
            4. `Python_Code_Executor` – To test code or logic.
            5. `summarize_tool` – For simplifying long outputs.

            ## Instructions:
            - Review the assistant's reasoning.
            - Use tools **only if needed** to verify accuracy or correctness.
            - Assess whether tools were used properly by the assistant.
            - Be fair and critical, but do not fix or improve responses.

            ## Output:
            - eval_feedback: `True` if the assistant response is factually and logically valid.
            - eval_feedback: `False` if incorrect, incomplete, or misuses tools.
            - reason: Short justification of your evaluation.
            """)

        evaluator_messages = [eval_prompt] + state["messages"]
        result: Evaluator = self.evaluator_with_tools.invoke(evaluator_messages)
        print(f"🧾 Evaluator Result: {result.eval_feedback}, Reason: {result.reason}")

        return {
            **state,
            "evaluator_response": result.reason,
            "eval_feedback": result.eval_feedback,
            "feedback": result.eval_feedback,  
        }

        
    def user_feedback(self, state: State) -> State:
        """Get user feedback"""
        print(f"\n🤖 Assistant Response: {state.get('llm_response', '')}")
        print(f"✅ Session so far")
        print(f"📊 Retry count: {state.get('retry_count', 0)}")
        print(f"👍 Feedback: {'Positive' if state.get('feedback', False) else 'Negative'}")
        user_feedback = input("AI: Is the llm response correct: say(y/n):")
        feedback = user_feedback.lower() == 'y'
        
        clarified_query = None
        retry_count = state.get('retry_count', 0)
        
        if not feedback and retry_count < 3:
            print("AI: Can you please rephrase your query more clearly?")
            clarified_query = input("User: ")
        
        return {
            **state,
            "feedback": feedback,
            "retry_count": retry_count + 1,
            "clarified_query": clarified_query,
        }
    
    def user_feedback_router(self, state: State):
        if state.get('feedback', False):
            return "end"
        else:
            return "continue"
        
    def evaluator_router(self, state: State):
        tool_use = state.get('tool_use', False)
        print("🔁 Evaluator router decision based on tool use...")
        print(f"➡️ Routing to {'evaluation_with_tools' if state.get('tool_use', False) else 'user_feedback'}")

        if tool_use:
            return "evaluation_with_tools"
        else:
            return "user_feedback"

def create_initial_state():
    """Create initial state with proper defaults"""
    return {
        "messages": [],
        "feedback": False,
        "eval_feedback": False,
        "retry_count": 0,
        "clarified_query": None,
        "llm_response": None,
        "evaluator_response": None,
        "tool_use": False,
    }

# Initialize agent
schema = Agent()
schema.setup()

# Build graph
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("llm", schema.agent)
graph_builder.add_node("tools", schema.tool_node)
graph_builder.add_node("evaluation", schema.evaluate)
graph_builder.add_node("evaluation_with_tools", schema.evaluate_with_tools)
graph_builder.add_node("user_feedback", schema.user_feedback)

# Edges
graph_builder.add_edge(START, "llm")

graph_builder.add_conditional_edges(
    "llm",
    schema.llm_router,
    {
        "tool_used": "tools",
        "evaluation": "evaluation"
    }
)

graph_builder.add_edge("tools", "llm")

graph_builder.add_conditional_edges(
    "evaluation",
    schema.evaluator_router,
    {
        "evaluation_with_tools": "evaluation_with_tools",
        "user_feedback": "user_feedback"
    }
)

graph_builder.add_conditional_edges(
    "user_feedback",
    schema.user_feedback_router,
    {
        "end": END,
        "continue": "llm"
    }
)

graph_builder.add_conditional_edges(
    "evaluation_with_tools",
    schema.user_feedback_router,
    {
        "end": END,
        "continue": "llm"
    }
)

# Final compilation
graph = graph_builder.compile(checkpointer=MemorySaver())

def run(user_input: str) -> dict:
    """Simple run function for the agent with proper configuration"""
    print("🧭 Starting graph execution...")
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "feedback": False,
        "eval_feedback": False,
        "retry_count": 0,
        "clarified_query": None,
        "llm_response": None,
        "evaluator_response": None,
        "tool_use": False,
    }
    
    # Configuration for LangGraph checkpointer
    config = {
        "configurable": {
            "thread_id": "default_thread"
        }
    }
    
    try:
        final_state = graph.invoke(initial_state, config=config)
        return {
            "success": True,
            "final_response": final_state.get("llm_response", ""),
            "retry_count": final_state.get("retry_count", 0),
            "feedback_received": final_state.get("feedback", False),
            "tool_used": final_state.get("tool_use", False)
        }

        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    print("🤖 Simple Agent with ToolNode")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("Enter your query: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye! 👋")
            break
        
        if not user_input:
            continue
        
        result = run(user_input)
        
        if result["success"]:
            print(f"\n🤖 Assistant Response: {result['final_response']}")
            print(f"✅ Session completed!")
            print(f"📊 Retry count: {result['retry_count']}")
            print(f"👍 Feedback: {'Positive' if result['feedback_received'] else 'Negative'}")
            print(f"🛠️ Tool used: {'Yes' if result.get('tool_used') else 'No'}")

        else:
            print(f"❌ Error: {result['error']}")
        
        print("-" * 50)