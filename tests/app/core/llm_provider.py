# src/core/llm_provider.py

from langchain_community.chat_models import ChatOllama
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMProvider:
    """
    Manages and provides initialized ChatOllama instances for the agents.
    
    This centralizes LLM configuration, simplifying model switching and ensuring 
    the Multi-LLM setup is enforced.
    """
    
    def __init__(self):
        # Base URL for the local Ollama server, usually http://localhost:11434
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Models assigned based on complexity/task
        self.LLAMA3_MODEL = os.getenv("LLAMA3_MODEL", "llama3")
        self.MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral")
        
        # Configuration for deterministic (reasoning) vs. creative (generation) tasks
        self.REASONING_CONFIG = {"temperature": 0.0, "base_url": self.ollama_base_url}
        self.CREATIVE_CONFIG = {"temperature": 0.5, "base_url": self.ollama_base_url}
        
        # Initialize providers
        self.reasoning_llm = self._create_llm(self.LLAMA3_MODEL, self.REASONING_CONFIG)
        self.creative_llm = self._create_llm(self.MISTRAL_MODEL, self.CREATIVE_CONFIG)
        
    def _create_llm(self, model_name: str, config: dict) -> ChatOllama:
        """Helper method to instantiate ChatOllama."""
        try:
            return ChatOllama(model=model_name, **config)
        except Exception as e:
            print(f"Error initializing LLM {model_name}: {e}")
            raise

    def get_reasoning_llm(self) -> ChatOllama:
        """Returns the Llama 3 instance (low temperature) for Manager/Sourcing/Pricing Agents."""
        return self.reasoning_llm

    def get_creative_llm(self) -> ChatOllama:
        """Returns the Mistral instance (medium temperature) for Listing/Order Agents."""
        return self.creative_llm

# --- Example of .env.example content ---
# OLLAMA_BASE_URL=http://localhost:11434
# LLAMA3_MODEL=llama3
# MISTRAL_MODEL=mistral