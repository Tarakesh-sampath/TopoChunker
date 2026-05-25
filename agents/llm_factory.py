import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from utils.config_loader import load_llm_config

def create_llm(
    model_type: Literal["openai", "deepseek"] = "deepseek",
    model_name: str = "deepseek-chat",
    temperature: float = 0.1,
) -> BaseChatModel:
    """
    Creates and returns an LLM instance using API (OpenAI or DeepSeek).

    优先从 `config/llm_config.json` 读取本地配置（不应提交到 git），
    若配置缺失则回退到环境变量。
    """
    config = load_llm_config()

    if model_type == "openai":
        api_key = (
            (config.get("openai") or {}).get("api_key")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        print(f"🌐 Using OpenAI API: {model_name}")
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
        
    elif model_type == "deepseek":
        deepseek_cfg = config.get("deepseek") or {}
        api_key = (
            deepseek_cfg.get("api_key")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        base_url = deepseek_cfg.get("base_url") or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        
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
            max_tokens=4096,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'openai' or 'deepseek'")
