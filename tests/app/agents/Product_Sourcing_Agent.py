# app/agents/sourcing.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from typing import List
from pydantic import BaseModel, Field

# Ensure you have imported or defined the actual tools
# NOTE: In a final project, you would import these from the tools module:
# from app.tools.data_tools import read_catalog_tool, write_json_output
# We use placeholder functions here for completeness.

# --- 1. Pydantic Schema for Structured Output ---
# This ensures the LLM's output is STRICTLY a list of 10 structured objects.

class ProductSelection(BaseModel):
    """Schema for a single selected product record."""
    supplier_sku: str = Field(description="The unique SKU selected from the catalog.")
    category: str = Field(description="The category of the selected product.")
    cost_price: float = Field(description="The original cost price of the product.")
    shipping_cost: float = Field(description="The shipping cost for the product.")
    reasoning: str = Field(description="Brief business justification (2-3 sentences) explaining why this SKU was selected over others to meet the 25% margin and high-potential criteria.")

class SelectionList(BaseModel):
    """A list containing exactly 10 selected products."""
    selected_products: List[ProductSelection] = Field(description="A list containing exactly 10 high-potential SKUs.")


# --- 2. The Agent Class Definition ---

class ProductSourcingAgent:
    """
    Agent responsible for applying business logic to select the top 10 products.
    Uses Llama 3 for complex, qualitative reasoning and selection.
    """
    def __init__(self, llm_provider, tools):
        # Use the reasoning LLM (Llama 3, low temperature)
        self.llm = llm_provider.get_reasoning_llm()
        self.parser = JsonOutputParser(pydantic_object=SelectionList)
        self.tools = tools # The tools provided by the workflow, e.g., read_catalog_tool
        
    def create_agent_chain(self):
        """
        Creates the LangChain runnable that includes prompt, tools, and parser.
        """
        
        # --- 3. The Focused System Prompt (The Agent's Persona) ---
        
        system_prompt = (
            "You are the senior **Product Sourcing Agent**. Your sole mission is to analyze the "
            "provided supplier catalog data and select the **TOP 10** SKUs for immediate listing.\n\n"
            "**CONSTRAINTS:**\n"
            "1. **MUST** select exactly 10 SKUs.\n"
            "2. **All selected SKUs** must have been pre-filtered for stock availability (`stock >= 10`).\n"
            "3. **Prioritize** SKUs that are most likely to hit or exceed a **25% profit margin** after all fees, "
            "based on the provided 'cost_price' and 'shipping_cost'.\n"
            "4. **Apply qualitative business logic:** Choose products that show high market appeal or category viability.\n"
            "5. **ALWAYS** output the result as a single JSON object strictly following the 'SelectionList' schema."
        )

        # The prompt template guides the LLM through the process
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", 
             "Step 1: Use the 'read_catalog_tool' to get the available data for analysis.\n"
             "Step 2: Analyze the filtered data for margin potential, category viability, and product appeal.\n"
             "Step 3: Select the 10 best SKUs and generate the required 'reasoning' for each selection.\n"
             "Output ONLY the final JSON list of the 10 selected products."
             
            )
        ])

        # --- 4. LangChain Runnable Chain (Core Logic) ---

        # The chain binds the prompt, the LLM, the tools, and the output parser.
        agent_chain = (
            prompt
            | self.llm.bind(tools=self.tools) # Bind the list of available tools
            | self.parser # Ensures JSON output is validated against the schema
        )
        return agent_chain