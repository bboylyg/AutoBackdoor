# llm_openai.py

import os
import openai
from typing import List, Dict

# Set API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# Default model - using gpt-4o-mini for cost efficiency
DEFAULT_MODEL = "gpt-4o-mini"

def query_llm(
    history: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """
    Query OpenAI Chat API with conversation history.
    
    Args:
    - history: Conversation history list [{"role": "user"/"assistant", "content": [...] }]
    - model: Model name, e.g., "gpt-4o-mini" or "gpt-4"
    - temperature: Controls output randomness
    - max_tokens: Maximum tokens to generate

    Returns:
    - Model response string (str)
    """
    try:
        # Flatten content format
        messages = []
        for turn in history:
            if isinstance(turn["content"], list):
                full_text = ""
                for segment in turn["content"]:
                    if segment["type"] == "text":
                        full_text += segment["text"]
                messages.append({"role": turn["role"], "content": full_text})
            else:
                messages.append(turn)

        print(f"🤖 [LLM] Querying {model}...")
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return f"[LLM ERROR] {str(e)}"
