



from ast import List
from json import tool
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field
from setuptools import Command

from langchain_core.prompts import ChatPromptTemplate

  


class ManagerState(BaseModel):
    messages: List[Dict[str, Any]] = Field("Conversation history between the Manager Agent and sub-agents.")
    supplier_catalog: List[Dict[str, Any]] = Field("Full supplier catalog loaded by the Manager Agent.")
    orders: List[Dict[str, Any]] = Field("List of orders to be processed.")
    selected_skus: List[Dict[str, Any]] = Field("List of SKUs selected from sourcing.")
    output_dir: str = "/out"
    input_dir: str = "/data"
    path_catalog: str = "/data/supplier_catalog.csv"
    path_orders: str = "/data/orders.csv"

@tool
def handoff_to_subagents(
    agent_name: Literal["sourcing", "listing","routing", "pricing", "qa","reporter"],
    task_details: str
):
    """Assign task to a sub agents
    
    Args:
        agent_name (Literal["sourcing", "listing","routing", "pricing", "qa
        , "reporter"]): The name of the sub-agent to hand off the task to.
        task_details (str): Details of the task to be performed by the sub-agent.
        """
    
    update = {
        "task_details": task_details,
        "messages": [{
            "name": f"handoff_to_{agent_name}",
            "content": f"Task handed off to {agent_name}."
        }],
    }
    return Command(
        goto=f"{agent_name}_node",
        update=update
    )

class ManagerAgent:
    """
    Orchestrates the workflow by deciding which agent to run next based on the state.
    """
    def __init__(self, llm_provider, tools):
        # Use the reasoning LLM (Llama 3, low temperature)
        self.llm = llm_provider.get_reasoning_llm()
        self.tools = tools # The tools provided by the workflow

    def create_agent_chain(self):
        """
        Creates the LangChain runnable that uses the handoff_to_subagents tool.
        """
        system_prompt = (
            "You are the **Manager Agent**. Your job is to orchestrate the dropshipping operation "
            "workflow sequentially. Analyze the current state and determine the next logical step. "
            "You MUST use the 'handoff_to_subagents' tool to transition control to the next specialized agent."
            "\n\n**Current Workflow Stage:**"
            "1. Sourcing -> 2. Listing -> 3. Pricing -> 4. Routing -> 5. Reporting."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            # The Manager reads the history (messages) and decides the next step
            ("human", "Determine the next agent to run based on the overall project goal.")
        ])

        # The chain binds the prompt, the LLM, and the tool
        agent_chain = (
            prompt
            | self.llm.bind(tools=self.tools)
        )
        return agent_chain
        

