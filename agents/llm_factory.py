import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

def create_llm(
    model_type: Literal["openai", "deepseek"] = "deepseek",
    model_name: str = "deepseek-chat",
    temperature: float = 0.1,
) -> BaseChatModel:
    """
    Creates and returns an LLM instance using API (OpenAI or DeepSeek).
    """
    if model_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        print(f"🌐 Using OpenAI API: {model_name}")
        return ChatOpenAI(model=model_name, temperature=temperature)
        
    elif model_type == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        
        # Standardize deepseek model name
        if model_name in ["default", "gpt-4o-mini"]:
            model_name = "deepseek-chat"
            
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY or OPENAI_API_KEY not set.")
        
        print(f"🌐 Using DeepSeek API: {model_name}")
        return ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_tokens=4096
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'openai' or 'deepseek'")
