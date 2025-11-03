# app/tools/data_tools.py

import pandas as pd
import json
import os
from langchain_core.tools import tool
from typing import Dict, Any

# --- Tool 1: Reading and Filtering Catalog Data ---

@tool
def read_catalog_tool(file_path: str) -> str:
    """
    Reads the supplier catalog CSV, performs deterministic filtering (stock >= 10),
    and returns a truncated, clean string for the LLM to analyze.
    
    The output is limited (e.g., to 30 SKUs) to save context window space while
    still allowing the LLM to perform qualitative selection.
    """
    try:
        # Check if file exists before trying to read
        if not os.path.exists(file_path):
            return f"Error: Catalog file not found at {file_path}"
            
        df = pd.read_csv(file_path)
        
        # 1. Select and format key columns needed for the LLM's selection reasoning
        df_for_llm = df[[
            'supplier_sku', 
            'name', 
            'category', 
            'cost_price', 
            'stock', 
            'shipping_cost'
        ]]
        
        # 2. PERFORM DETERMINISTIC FILTERING: Stock >= 10 (as required by the brief)
        filtered_df = df_for_llm[df_for_llm['stock'] >= 10].copy()
        
        if filtered_df.empty:
            return "Catalog Data: No products found that meet the minimum stock requirement (>= 10)."
        
        # 3. Truncate and convert to a string (CSV format is easy for LLMs to parse)
        # Limiting to .head(30) prevents token overload if thousands of products are eligible
        return (
            f"Catalog Data (Eligible SKUs with Stock >= 10, First {len(filtered_df.head(30))} rows):\n"
            f"{filtered_df.head(30).to_csv(index=False)}"
        )

    except Exception as e:
        return f"Error reading catalog file: {str(e)}"

# --- Tool 2: Writing Final Output ---

@tool
def write_json_output(data: Dict[str, Any], output_path: str) -> str:
    """
    Writes a dictionary or Pydantic object (converted to dict) to a JSON file 
    at the specified output path. Used for saving the final selection list.
    """
    try:
        # Ensure the directory structure exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Handle Pydantic objects by converting them to a dictionary if necessary
        if hasattr(data, 'model_dump'):
            data = data.model_dump()
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        # The tool MUST return a success confirmation string to the LLM
        return f"SUCCESS: JSON data written to {output_path}"
        
    except Exception as e:
        return f"ERROR: Could not write JSON file to {output_path}: {str(e)}"
    



# --- Configuration (Based on Project Requirement) ---
PLATFORM_FEE_RATE = 0.029  # 2.9%
PLATFORM_FEE_FIXED = 0.30  # $0.30
GST_RATE = 0.10            # 10% (for AU only, assuming AU operations for formula)
MIN_MARGIN = 0.25          # 25%

@tool
def calculate_minimum_price(cost_price: float, shipping_cost: float) -> Dict[str, float]:
    """
    Calculates the minimum required selling price (P) for a product to achieve a 
    minimum 25% margin, factoring in all fees (Platform, GST).
    The price is rounded up to the nearest $0.50.
    
    Returns: A dict with 'min_price' and 'margin_percentage'.
    """
    import math # Import here to ensure the tool is self-contained
    
    # 1. Define Variables
    variable_costs_rate = PLATFORM_FEE_RATE + GST_RATE 
    fixed_costs = cost_price + shipping_cost + PLATFORM_FEE_FIXED
    target_margin = MIN_MARGIN
    
    # 2. Solve for Price (P)
    # Formula: P = fixed_costs / (1 - target_margin - variable_costs_rate)
    denominator = 1 - target_margin - variable_costs_rate
    
    if denominator <= 0:
        # Should not happen with current rates, but prevents division by zero.
        raise ValueError("Margin target and fees exceed 100%. Price calculation is impossible.")

    min_price_unrounded = fixed_costs / denominator
    
    # 3. Round Up to Nearest $0.50
    # math.ceil(x * 2) / 2.0 rounds up to the nearest 0.5
    final_price = math.ceil(min_price_unrounded * 2) / 2.0

    # 4. Recalculate Final Margin for Reporting
    total_cost = fixed_costs + (final_price * variable_costs_rate)
    final_profit = final_price - total_cost
    final_margin_percentage = (final_profit / final_price) if final_price > 0 else 0
    
    return {
        "recommended_price": round(final_price, 2),
        "margin_percentage": round(final_margin_percentage * 100, 2)
    }

# --- Tool for generating stock updates ---
# This is a simple data formatting tool, not an LLM tool.

@tool
def create_stock_update_data(sku: str, stock: int) -> Dict[str, Any]:
    """
    Formats stock data into the required CSV structure.
    """
    return {
        "supplier_sku": sku,
        "stock_level": stock,
        "action": "SYNC_UPDATE"
    }