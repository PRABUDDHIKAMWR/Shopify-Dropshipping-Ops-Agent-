# tests/test_sourcing_agent.py

import os
import json
import pytest
from app.core.llm_provider import LLMProvider
from app.agents.Product_Sourcing_Agent import ProductSourcingAgent, SelectionList
from app.tools.data_tools import read_catalog_tool, write_json_output

# --- Setup Fixture for Test Environment ---

# Define the path for the mock catalog file
MOCK_CATALOG_PATH = "data/mock_catalog.csv"
OUTPUT_DIR = "tests/temp_out"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_selection.json")

# Create a simple, fake catalog file for testing
@pytest.fixture(scope="session", autouse=True)
def setup_mock_data():
    """Ensures a mock catalog file exists and cleans up the output directory."""
    os.makedirs("data", exist_ok=True)
    
    # Create a CSV with products that meet and fail the stock criteria
    csv_content = """supplier_sku,name,category,cost_price,stock,shipping_cost
SKU001,Premium Widget,Electronics,10.00,50,5.00
SKU002,Cheap Gadget,Home Goods,2.00,5,1.00
SKU003,Luxury Lamp,Electronics,50.00,20,10.00
SKU004,Simple Mug,Home Goods,1.00,10,2.00
SKU005,Smart Watch Pro,Electronics,80.00,100,15.00
SKU006,Book Collection,Books,5.00,15,5.00
SKU007,High-End Stereo,Electronics,120.00,40,20.00
SKU008,Kitchen Blender,Home Goods,25.00,10,8.00
SKU009,T-Shirt Set,Apparel,7.00,60,3.00
SKU010,Designer Jeans,Apparel,40.00,12,7.00
SKU011,Basic Headphones,Electronics,15.00,50,5.00
SKU012,Garden Hose,Outdoor,10.00,50,5.00
SKU013,Coffee Maker,Home Goods,30.00,50,5.00
SKU014,VR Headset,Electronics,150.00,50,5.00
SKU015,Running Shoes,Apparel,60.00,50,5.00
# ... add more SKUs to reach a pool greater than 10 ...
"""
    # Write at least 10 products that meet the stock requirement
    with open(MOCK_CATALOG_PATH, "w") as f:
        f.write(csv_content * 2) # Ensure > 10 eligible SKUs

    # Clean up output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    yield 
    # Teardown (optional): Remove created files after all tests run
    # os.remove(MOCK_CATALOG_PATH)


# --- The Test Function ---

def test_sourcing_agent_selection():
    """
    Tests the Product Sourcing Agent's ability to select 10 SKUs, apply reasoning, 
    and output valid JSON.
    """
    # 1. Initialization
    llm_provider = LLMProvider()
    sourcing_tools = [read_catalog_tool, write_json_output]
    agent = ProductSourcingAgent(llm_provider, tools=sourcing_tools)
    agent_chain = agent.create_agent_chain()
    
    # 2. Define the initial state the agent will start with
    # Note: We simulate the initial data load that the Manager Agent normally performs.
    initial_state = {
        "catalog": MOCK_CATALOG_PATH, # The path is what the tool needs
        "orders": "data/mock_orders.csv", # Placeholder
        "output_dir": OUTPUT_DIR,
        "intermediate_steps": [], # Required for LangChain's tool use tracking
    }

    print("\n--- Invoking Sourcing Agent Chain ---")
    
    # 3. Execution
    # The agent will use its tools, reason, and produce a final JSON output
    # The output should be a Python dictionary/Pydantic object thanks to the JsonOutputParser
    try:
        selection_output = agent_chain.invoke(initial_state)
    except Exception as e:
        pytest.fail(f"Agent failed to execute or parse output: {e}")


    # 4. Verification

    # V1: Verify Output Structure (Pydantic validation)
    # This automatically verifies if the output is a dictionary/object matching the schema
    assert isinstance(selection_output, dict), "Output must be a dictionary (JSON object)."
    SelectionList(**selection_output) # Will raise ValidationError if structure is wrong
    
    # V2: Verify Core Content (10 SKUs and Reasoning)
    selected_products = selection_output.get("selected_products", [])
    
    assert len(selected_products) == 10, f"Agent MUST select exactly 10 SKUs. Found: {len(selected_products)}"
    
    for item in selected_products:
        assert "supplier_sku" in item, "Each selected item must contain 'supplier_sku'."
        assert "reasoning" in item and len(item['reasoning']) > 10, "Each item must have a detailed 'reasoning'."
        
    print("\n✅ Sourcing Agent Test Passed: Output is valid, contains 10 items, and includes reasoning.")

    # V3: Optional - Check File Persistence
    # If the agent calls write_json_output, check if the file exists
    # If the agent is built as a pure node in LangGraph, the next node/workflow may call write_json_output,
    # but for isolated testing, we rely on the returned object.