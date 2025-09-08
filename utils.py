"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
import os
from pathlib import Path
from dotenv import load_dotenv

"""Load .env file from project root directory."""
# Get the current file's directory (src/react_agent/)
current_file = Path(__file__)
project_root = current_file.parent  
# Navigate up to project root: src/react_agent/ -> src/ -> project_root/
env_file = project_root / ".env"
load_dotenv(env_file)

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)

    # For Azure OpenAI, verify required environment variables are present and pass additional parameters
    if provider == "azure_openai":
        # For Azure OpenAI, we need to pass the deployment name as azure_deployment
        # The model parameter is treated as the deployment name
        return init_chat_model(
            model_provider=provider,
            model=model,  # This will be used as azure_deployment
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

    return init_chat_model(model, model_provider=provider)

#print(load_chat_model("azure_openai/gpt-4o"))  # Example usage
#print(load_chat_model("openai/gpt-4o"))  # Example usage
