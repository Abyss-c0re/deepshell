import ollama
import asyncio
import numpy as np
from utils.logger import Logger
from typing import AsyncGenerator, Sequence, cast
from config.settings import Mode, MODE_CONFIGS, EMBEDDING_MODEL, DEFAULT_HOST

logger = Logger.get_logger()

class OllamaClient:
    # Class-level lock to ensure critical async functions do not run concurrently
    _global_lock = asyncio.Lock()

    def __init__(
            self,
            host:str,
            model:str, 
            config:dict, 
            mode:Mode, 
            stream:bool = True, 
            render_output:bool = True, 
            show_thinking:bool = False
    ):

        logger.info("Initializing OllamaClient")
        self.client = ollama.AsyncClient(host=host)
        self.model = model
        self.config = config
        self.mode = mode
        self.stream = stream

        self.pause_stream = False
        self.output_buffer = asyncio.Queue()
        self.render_output = render_output

        self.show_thinking = show_thinking
        self.thoughts = []

        self.last_response = ""

        self.keep_history = True

        logger.info(f"Client initialized with model: {model}, mode: {mode}, stream: {stream}")

    def switch_mode(
            self,
            mode:Mode
    ) -> None:
        """
        Dynamically switches mode and updates config.
        """
        logger.info(f"Switching mode from {self.mode} to {mode}")
        if mode == self.mode:
            logger.info("Mode is the same, no switch needed")
            return

        try:
            config = MODE_CONFIGS[mode]
            self.model = config["model"]
            self.config = {"temperature": config["temp"], "system": config["prompt"]}
            self.stream = config["stream"]
            self.mode = mode
            logger.info(f"Mode switched successfully: {self.mode}")
        except KeyError as e:
            logger.error(f"Invalid mode: {mode}. Error: {e}")

    async def _chat_stream(
            self, 
            input=None, 
            history=None
    ) -> None:
        """
        Fetches response from the Ollama API and streams into output buffer.
        """
        async with OllamaClient._global_lock:
            logger.info(f"{self.mode.name} started stream")

            if history:
                input = history
            else:
                input = [{"role": "user", "content": input}]

            logger.debug(f"Chat request payload: {input}")

            try:
                # Force-cast the response to an AsyncGenerator
                response = cast(AsyncGenerator[dict, None], await self.client.chat(
                    model=self.model,
                    messages=input,
                    options=self.config,
                    stream=self.stream
                ))

                async for part in response:
                    if not self.pause_stream:
                        content = part.get('message', {}).get('content', '') 
                        await self.output_buffer.put(content)

                if not self.pause_stream:
                    await self.output_buffer.put(None)
                    logger.info("Chat stream ended successfully")

            except Exception as e:
                logger.error(f"Error during chat stream: {e}")

    async def _describe_image(
            self, 
            image: str | None, 
            prompt: str = "Describe"
    )-> str | None:
        """
        Describes an image using the vision model.
        """
        async with OllamaClient._global_lock:
            logger.info(f"{self.mode.name} describing image")
            
            if not image:
                logger.warning("No image provided")
                return "No image provided"

            if self.mode == Mode.VISION: 
                try:
                    response = await self.client.generate(model=self.model, prompt=prompt, images = [image])
                    logger.debug(f"Image description response: {response}")
                    message_data = response.response

                    if message_data:
                        return message_data
                    else:
                        logger.warning("No message found in response")
                        return "No message in response"

                except Exception as e:
                    logger.error(f"Error while describing image: {e}")
                    return "Error processing image"

    async def _fetch_response(
            self, 
            input:str
    ) -> str:
        """
        Fetches a complete response from the model.
        """
        async with OllamaClient._global_lock:
            logger.info(f"{self.mode.name} is fetching response")

            try:
                response = await self.client.generate(model=self.model, prompt=input)
                logger.info("Response received successfully")
                message_data = response.response
                if not message_data:
                    logger.warning("No message found in response")
                    return "No message in response"

                return message_data 
            except Exception as e:
                logger.error(f"Error fetching response: {e}")
                return "Error fetching response"

    async def _call_function(
            self, 
            input:str, 
            functions: list  = []
    )-> Sequence | None:
        """
        Fetches a complete response from the model.
        """
        async with OllamaClient._global_lock:
            logger.info(f"{self.mode.name} is fetching response")
            logger.info(functions)

            try:
                message = {'role': 'user', 'content': input}
                response = await self.client.chat(model=self.model, messages=[message],tools = functions)
                logger.info(f"Full response: {response}")
                if response.message.tool_calls:
                    return response.message.tool_calls
                             
                else:
                    logger.info("No functions have been called")
                    return None
                
            except Exception as e:
                logger.error(f"Error fetching response: {e}")
                return "Error fetching response"

    @staticmethod
    async def fetch_embedding(
            text: str
    )-> np.ndarray | None:
        """
        Asynchronously fetches and caches an embedding for the given text while ensuring 
        that no other locked operation (such as streaming) runs concurrently.
        """
        async with OllamaClient._global_lock:
            client = ollama.Client(host=DEFAULT_HOST)
            try:
                logger.info("Fetching embedding")
                # Offload blocking work to a thread if needed
                response = await asyncio.to_thread(client.embeddings, model=EMBEDDING_MODEL, prompt=text)
                embedding = response['embedding']
                logger.debug(f"Extracted {len(embedding)} embeddings")
                return embedding
            except Exception as e:
                logger.error(f"Error fetching embedding for text: {text}. Error: {str(e)}")
                return

