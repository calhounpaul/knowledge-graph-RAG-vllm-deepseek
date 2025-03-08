"""
DeepSeek Chat Module

location root/libs/query_llm.py

This module provides functionality to interact with DeepSeek models via the Hyperbolic API.
It supports both synchronous and streaming conversations with configurable parameters.
"""

import os
import logging
from typing import List, Dict, Union, Generator, Optional
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

this_dir = os.path.dirname(os.path.abspath(__file__))

class DeepSeekChat:
    """
    A client for interacting with DeepSeek models via the Hyperbolic API.
    """
    
    def __init__(
        self,
        api_key_path: str = os.path.join(this_dir, "hyperbolic_api_key.txt"),
        api_base: str = "https://api.hyperbolic.xyz/v1/",
        model_name: str = "deepseek-ai/DeepSeek-V3",
        max_tokens: int = 16000,
        temperature: float = 0.7,
        top_p: float = 0.95
    ):
        """
        Initialize the DeepSeek chat client.
        
        Args:
            api_key_path: Path to the file containing the Hyperbolic API key
            api_base: Base URL for the Hyperbolic API
            model_name: Name of the DeepSeek model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
        """
        self.api_key_path = api_key_path
        self.api_base = api_base
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Load the API key
        self._load_api_key()
        
        # Initialize the client
        self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        
    def _load_api_key(self) -> None:
        """
        Load the Hyperbolic API key from the specified file.
        
        Raises:
            FileNotFoundError: If the API key file doesn't exist
        """
        try:
            with open(self.api_key_path, "r") as f:
                self.api_key = f.read().strip()
            logger.debug("API key loaded successfully")
        except FileNotFoundError:
            logger.error(f"API key file not found at {self.api_key_path}")
            raise
    
    def _format_messages(self, messages_or_prompt: Union[List[Dict[str, str]], str]) -> List[Dict[str, str]]:
        """
        Format the input as a list of message dictionaries.
        
        Args:
            messages_or_prompt: Either a list of message dictionaries or a string prompt
            
        Returns:
            A list of message dictionaries in the format expected by the API
        """
        if isinstance(messages_or_prompt, str):
            return [{"role": "user", "content": messages_or_prompt}]
        return messages_or_prompt
    
    def chat(
        self,
        messages_or_prompt: Union[List[Dict[str, str]], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Send a chat request to the DeepSeek model and get the full response.
        
        Args:
            messages_or_prompt: Either a list of message dictionaries or a string prompt
            max_tokens: Override the default max_tokens setting
            temperature: Override the default temperature setting
            top_p: Override the default top_p setting
            
        Returns:
            The generated response as a string
        """
        messages = self._format_messages(messages_or_prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                top_p=top_p or self.top_p,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
    
    def chat_stream(
        self,
        messages_or_prompt: Union[List[Dict[str, str]], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Generator[str, None, None]:
        """
        Send a chat request to the DeepSeek model and stream the response.
        
        Args:
            messages_or_prompt: Either a list of message dictionaries or a string prompt
            max_tokens: Override the default max_tokens setting
            temperature: Override the default temperature setting
            top_p: Override the default top_p setting
            
        Yields:
            Chunks of the generated response as they become available
        """
        messages = self._format_messages(messages_or_prompt)
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                top_p=top_p or self.top_p,
                stream=True
            )
            
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    yield delta
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}")
            raise

    def conversation(self, system_prompt: Optional[str] = None) -> "Conversation":
        """
        Create a new conversation object that maintains message history.
        
        Args:
            system_prompt: Optional system prompt to set the behavior of the model
            
        Returns:
            A Conversation object
        """
        return Conversation(self, system_prompt)


class Conversation:
    """
    A conversation with a DeepSeek model that maintains message history.
    """
    
    def __init__(self, client: DeepSeekChat, system_prompt: Optional[str] = None):
        """
        Initialize a conversation.
        
        Args:
            client: DeepSeekChat client to use for API calls
            system_prompt: Optional system prompt to set the behavior of the model
        """
        self.client = client
        self.messages = []
        
        # Add system prompt if provided
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ("user", "assistant", or "system")
            content: The content of the message
        """
        self.messages.append({"role": role, "content": content})
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.
        
        Args:
            content: The content of the user message
        """
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation.
        
        Args:
            content: The content of the assistant message
        """
        self.add_message("assistant", content)
    
    def get_response(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Get a response from the model based on the conversation history.
        The response is automatically added to the conversation history.
        
        Args:
            max_tokens: Override the default max_tokens setting
            temperature: Override the default temperature setting
            top_p: Override the default top_p setting
            
        Returns:
            The generated response as a string
        """
        response = self.client.chat(
            self.messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Add the response to the conversation history
        self.add_assistant_message(response)
        
        return response
    
    def get_stream_response(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Generator[str, None, None]:
        """
        Get a streaming response from the model based on the conversation history.
        The complete response is automatically added to the conversation history
        after the stream is exhausted.
        
        Args:
            max_tokens: Override the default max_tokens setting
            temperature: Override the default temperature setting
            top_p: Override the default top_p setting
            
        Yields:
            Chunks of the generated response as they become available
        """
        full_response = ""
        
        for chunk in self.client.chat_stream(
            self.messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        ):
            full_response += chunk
            yield chunk
        
        # Add the complete response to the conversation history
        self.add_assistant_message(full_response)
    
    def reset(self, keep_system_prompt: bool = True) -> None:
        """
        Reset the conversation history.
        
        Args:
            keep_system_prompt: Whether to keep the system prompt if one exists
        """
        if keep_system_prompt and self.messages and self.messages[0]["role"] == "system":
            system_prompt = self.messages[0]
            self.messages = [system_prompt]
        else:
            self.messages = []


# Example usage
if __name__ == "__main__":
    # Create a client
    client = DeepSeekChat()
    
    # Simple single-turn query
    response = client.chat("What are the key features of Python?")
    print(f"Response: {response}")
    
    # Create a conversation with a system prompt
    conversation = client.conversation(
        system_prompt="You are a helpful assistant specialized in programming."
    )
    
    # Add a user message and get a response
    conversation.add_user_message("How do I read a file in Python?")
    response = conversation.get_response()
    print(f"Response: {response}")
    
    # Continue the conversation
    conversation.add_user_message("And how do I write to a file?")
    
    # Stream the response
    print("\nStreaming response:")
    for chunk in conversation.get_stream_response():
        print(chunk, end="", flush=True)
    print("\n")
    
    # Print the conversation history
    print("\nConversation History:")
    for msg in conversation.messages:
        print(f"{msg['role']}: {msg['content'][:50]}...")