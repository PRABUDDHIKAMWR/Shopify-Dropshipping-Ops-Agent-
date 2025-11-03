# --- Inside app/workflow/ops_graph.py ---
import json
import os
from app.agents.Product_Sourcing_Agent import ProductSourcingAgent
from app.tools.data_tools import read_catalog_tool, write_json_output
from app.core.llm_provider import LLMProvider
from tests.app.agents.Listing_Agent import ListingAgent
from tests.app.agents.Manager_Agent import ManagerAgent, ManagerState
from langchain_core.runnables.config import RunnableConfig

# Create tool list
sourcing_tools = [read_catalog_tool, write_json_output]

# Initialize the agent
sourcing_agent_instance = ProductSourcingAgent(LLMProvider, tools=sourcing_tools)


def manager_node(state: ManagerState, agent_instance: ManagerAgent) -> str:
    """
    LangGraph node function for the Manager Agent (determines the next transition).
    """
    print("\n--- Running Node: Manager Agent (Decision Maker) ---")

    # 1. Execute the Manager Agent's Chain
    # The Manager LLM reads the state's messages and decides to use the tool.
    manager_chain = agent_instance.create_agent_chain()
    
    # The output from the LLM is expected to be a tool call (handoff_to_subagents)
    # The Manager Agent's tool returns a Command object
    manager_chain.invoke(state)



def sourcing_node(state:ManagerState, config: RunnableConfig):
    """call the sourcing agent to select products from the catalog"""

    result = sourcing_agent_instance.invoke(
        {
            "supplier_catalog": state.supplier_catalog
        },
        config=config
    )
    sourcing_path = os.path.join(state["output_dir"], "selection.json")
    write_json_output(result.selected_products, sourcing_path)

    state.selected_skus = result.selected_products
    state.messages.append({
        "name":"sourcing_agent",
        "content":"Selected SKUs from sourcing agent."
    })
    return state


def listing_node(state: ManagerState, agent_instance: ListingAgent):
    """
    LangGraph node function for the Listing Agent, which takes the 10 selected SKUs
    and generates content for all of them.
    """
    print("\n--- Running Node: Listing Agent (Content Generation) ---")
    
    selected_skus = state.get("selected_skus")
    
    if not selected_skus or len(selected_skus) != 10:
        print("ERROR: Listing Agent received fewer than 10 selected SKUs. Aborting node.")
        # Return state unchanged or raise error if critical
        return {}

    # 1. Prepare Input: The LLM works best when given a clean, structured list.
    # We strip down the input data to only the fields needed for creative content.
    input_data = [
        {
            "supplier_sku": sku["supplier_sku"],
            "name": sku.get("name"),
            "category": sku.get("category"),
            "description_snippet": sku.get("description", "")[:200], # Truncate description if too long
            "brand": sku.get("brand"),
        }
        for sku in selected_skus
    ]
    
    # 2. Execute the Generation Chain
    agent_chain = agent_instance.create_generation_chain()
    
    # The agent is called once with the JSON representation of all 10 products
    result= agent_chain.invoke({
        "product_data_json": json.dumps(input_data)
    })
    
    # 3. Save Output Artifact
    listing_path = os.path.join(state["output_dir"], "listings.json")
    write_json_output(result, listing_path)

    state.messages.append({
        "name":"listing_agent",
        "content":f"Generated content for {len(result['listings'])} products."
    })
    return state




