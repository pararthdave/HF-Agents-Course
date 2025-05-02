from smolagents import (
    CodeAgent,
    InferenceClientModel,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    HfApiModel,
    LiteLLMModel,
    tool
)
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
import os
load_dotenv()
model_id = "meta-llama/Llama-3.3-70B-Instruct" 

@tool
def basic_inference(
    prompt: str,
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
    provider: str = "groq",
) -> str:
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

@tool
def toolcalling(
    prompt: str,
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
    provider: str = "groq",
) -> str:
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

@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo for a query and return maximum 3 result.

    Args:
        query: The search query."""
    search_tool = DuckDuckGoSearchTool()
    search_docs = search_tool(query)
    return search_docs


class BotMan:
    def __init__(self,
        model_type: str = "HfApiModel",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = 0,
        additional_imports: List[str] = None,
        executor_type: str = "local",
                 ):
        """
        Initialize the BotMan class.
        """
        if model_type == "HfApiModel":
            if api_key is None:
                api_key = os.environ.get("HUGGINGFACEHUB_API_KEY")
                if not api_key:
                    raise ValueError("API key is required for HfApiModel.")
        self.model = InferenceClientModel(
            model_id=model_id or "meta-llama/Llama-3.3-70B-Instruct",
            token=api_key,
            provider=provider or "hf-inference",
            temperature=temperature,
            timeout=timeout or 80
        )
        self.tools = [
            web_search,
            basic_inference,
            toolcalling,
        ]
        executor_kwargs = {}
        self.imports = ["pandas", "numpy", "datetime", "json", "re", "math", "os", "requests", "csv", "urllib"]
        if additional_imports:
            self.imports.extend(additional_imports)

        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            additional_authorized_imports=self.imports,
            executor_type=executor_type,
            executor_kwargs=executor_kwargs,
        )

    def answer(self, question: str) -> str:
        """
        Answer a question using the agent.
        """
        try:
            result = self.agent.run(question)
            return result
        except Exception as e:
            print(f"Error during inference: {e}")
            return str(e)

if __name__ == '__main__':
    # Example usage
    bot = BotMan(model_type="HfApiModel", model_id=model_id, api_key=os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    question = "What is the capital of France?"
    answer = bot.answer(question)
    print(answer)