from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict, Any
from pydantic import BaseModel, Field
 
# Pydantic Schemas for Structured Output ---

class ListingContent(BaseModel):
    """Schema for the generated content of a single product listing."""
    supplier_sku: str = Field(description="The unique SKU for this product listing.")
    shopify_title: str = Field(description="An engaging, SEO-optimized title (max 60 chars).")
    key_bullets: List[str] = Field(description="A list of 3-5 compelling, benefit-focused bullet points.")
    description_html: str = Field(description="A detailed, marketing-focused description formatted using basic HTML tags (p, ul, li, strong).")
    seo_tags: List[str] = Field(description="A list of 5-8 relevant SEO keywords/tags for the product.")

class ListingOutput(BaseModel):
    """The final list of all generated listings."""
    listings: List[ListingContent] = Field(description="A list containing the listing content for all 10 selected SKUs.")


# --- 2. The Agent Class Definition ---

class ListingAgent:
    """
    Agent responsible for generating high-quality, SEO-optimized content for Shopify listings.
    Uses the Mistral model for creative and persuasive output.
    """
    def __init__(self, llm_provider, tools):
        # Use the creative LLM (Mistral, higher temperature)
        self.llm = llm_provider.get_creative_llm()
        self.parser = JsonOutputParser(pydantic_object=ListingOutput)
        self.tools = tools # e.g., write_json_output tool
        
    def create_generation_chain(self):
        """
        Creates the LangChain runnable for content generation.
        """
        
        # --- 3. The Creative System Prompt (The Agent's Persona) ---
        
        system_prompt = (
            "You are the **Shopify Listing Copywriter**. Your goal is to take raw product data and "
            "transform it into highly compelling, SEO-optimized marketing copy ready for e-commerce.\n"
            "Focus on the benefits, not just the features. Inject enthusiasm and clarity. "
            "You MUST output the result as a single JSON object strictly following the 'ListingOutput' schema."
        )

        # The prompt template uses the 'product_data_json' variable passed in the invoke call
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", 
             "Generate the full listing content (Title, Bullets, HTML Description, SEO Tags) "
             "for the following list of products, which are provided as a single JSON array:\n\n"
             "{product_data_json}\n\n"
             "Ensure the description_html uses proper HTML formatting and lists all required fields."
            )
        ])

        # --- 4. LangChain Runnable Chain ---

        agent_chain = (
            prompt
            | self.llm
            | self.parser # Validates and parses the JSON output
        )
        return agent_chain