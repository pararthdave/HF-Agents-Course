from smolagents import CodeAgent, InferenceClientModel, ToolCallingAgent, DuckDuckGoSearchTool
from dotenv import load_dotenv
import os
load_dotenv()
model_id = "meta-llama/Llama-3.3-70B-Instruct" 


def basic_inference(
    prompt: str,
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
    provider: str = "groq",
):
    """
    Run a basic inference using the specified model and provider.

    Args:
        prompt (str): The input prompt for the model.
        model_id (str): The ID of the model to use.
        provider (str): The provider to use for inference.

    Returns:
        str: The model's response.
    """
    # Load the model
    model = InferenceClientModel(model_id=model_id, token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")) # You can choose to not pass any model_id to InferenceClientModel to use a default model
    # you can also specify a particular provider e.g. provider="together" or provider="sambanova"

    # Create an agent with the specified tools and model
    agent = CodeAgent(tools=[], model=model, add_base_tools=True)

    # Run the agent with the provided prompt
    return agent.run(prompt)

def toolcalling(
    prompt: str,
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
    provider: str = "groq",
):
    """
    Run a tool calling inference using the specified model and provider.

    Args:
        prompt (str): The input prompt for the model.
        model_id (str): The ID of the model to use.
        provider (str): The provider to use for inference.

    Returns:
        str: The model's response.
    """
    # Load the model
    model = InferenceClientModel(model_id=model_id, token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")) # You can choose to not pass any model_id to InferenceClientModel to use a default model
    # you can also specify a particular provider e.g. provider="together" or provider="sambanova"

    # Create an agent with the specified tools and model
    agent = ToolCallingAgent(tools=[], model=model, add_base_tools=True)

    # Run the agent with the provided prompt
    return agent.run(prompt)

def web_search(query: str) -> str:
    """Search DuckDuckGo for a query and return maximum 3 result.

    Args:
        query: The search query."""
    search_tool = DuckDuckGoSearchTool()
    search_docs = search_tool(query)
    return search_docs