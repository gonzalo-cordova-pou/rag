from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AgentState:
    """
    Represents the state of the AI agent during execution.

    Attributes:
        messages (List[Dict[str, Any]]): List of messages in the conversation
        next_steps (List[str]): List of next steps to be taken in the workflow
    """
    def __init__(self, messages: List[Dict[str, Any]] = None, next_steps: List[str] = None):
        """
        Initialize the agent state.

        Args:
            messages (List[Dict[str, Any]], optional): Initial messages. Defaults to empty
                list.
            next_steps (List[str], optional): Initial next steps. Defaults to empty list.
        """
        self.messages = messages or []
        self.next_steps = next_steps or []


class AIAgent:
    """
    A modular AI agent that processes user requests through a structured workflow.

    This agent uses LangGraph to create a workflow with three main nodes:
    1. Parse Request: Understands the user's input
    2. Think: Plans how to solve the task
    3. Execute: Provides the solution

    Attributes:
        llm (ChatOpenAI): The language model used by the agent
        workflow (StateGraph): The graph defining the agent's workflow
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize the AI agent.

        Args:
            model_name (str): Name of the language model to use
            temperature (float): Temperature parameter for the language model
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.workflow = self._create_workflow()

    def _parse_request(self, state: AgentState) -> AgentState:
        """Parse the user request and determine the task"""
        user_input = state.messages[-1]["content"]
        prompt = f"Parse this user request into a clear task: {user_input}"
        response = self.llm.invoke(prompt)
        state.messages.append(
            {"role": "system", "content": f"Parsed task: {response.content}"}
        )
        return state

    def _think(self, state: AgentState) -> AgentState:
        """Think about how to solve the task"""
        task = state.messages[-1]["content"]
        prompt = f"Think step by step about how to solve this task: {task}"
        response = self.llm.invoke(prompt)
        state.messages.append(
            {"role": "system", "content": f"Thinking process: {response.content}"}
        )
        return state

    def _execute(self, state: AgentState) -> AgentState:
        """Execute the solution plan"""
        thinking = state.messages[-1]["content"]
        prompt = f"Based on this thinking, provide a solution: {thinking}"
        response = self.llm.invoke(prompt)
        state.messages.append({"role": "assistant", "content": response.content})
        return state

    def _decide_next_step(self, state: AgentState) -> str:
        """Decide the next step in the workflow"""
        if len(state.messages) <= 1:
            return "parse"
        elif len(state.messages) == 2:
            return "think"
        else:
            return "execute"

    def _create_workflow(self) -> StateGraph:
        """Create and configure the agent's workflow graph"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("parse", self._parse_request)
        workflow.add_node("think", self._think)
        workflow.add_node("execute", self._execute)

        # Add edges
        workflow.add_conditional_edges(START, self._decide_next_step)
        workflow.add_edge("parse", "think")
        workflow.add_edge("think", "execute")
        workflow.add_edge("execute", END)

        return workflow.compile()

    def invoke(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user input through the agent's workflow.

        Args:
            user_input (str): The user's input message

        Returns:
            Dict[str, Any]: The final state of the agent after processing
        """
        initial_state = AgentState(
            messages=[{"role": "user", "content": user_input}]
        )
        return self.workflow.invoke(initial_state)

    def get_response(self, result: Dict[str, Any]) -> str:
        """
        Extract the final response from the agent's result.

        Args:
            result (Dict[str, Any]): The result from invoking the agent

        Returns:
            str: The final response message
        """
        return result["messages"][-1]["content"]


# Example usage
if __name__ == "__main__":
    # Create an instance of the AI agent
    agent = AIAgent()

    # Process a user request
    result = agent.invoke("Find the best restaurants in New York")

    # Get and print the response
    print(agent.get_response(result))
