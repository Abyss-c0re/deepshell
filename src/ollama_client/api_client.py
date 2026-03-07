import openai
import asyncio
import numpy as np
from src.utils.logger import Logger
from typing import Sequence
from src.config.settings import Mode, MODE_CONFIGS, EMBEDDING_MODEL, DEFAULT_HOST

logger = Logger.get_logger()


class LLMClient:
    # Class-level lock to ensure critical async functions do not run concurrently
    _global_lock = asyncio.Lock()

    def __init__(
        self,
        host: str,
        model: str,
        config: dict,
        mode: Mode,
        stream: bool = True,
        render_output: bool = True,
        show_thinking: bool = False,
    ):
        logger.info("Initializing LLMClient")
        self.client = openai.AsyncOpenAI(base_url=host + "/v1", api_key="sk-no-key-required")
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

        logger.info(
            f"Client initialized with model: {model}, mode: {mode}, stream: {stream}"
        )

    def switch_mode(self, mode: Mode) -> None:
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

    async def _chat_stream(self, input=None, history=None) -> None:
        """
        Fetches response from the llama.cpp API and streams into output buffer.
        """
        async with LLMClient._global_lock:
            logger.info(f"{self.mode.name} started stream")

            if history:
                messages = history
            else:
                messages = [
                    {"role": "system", "content": self.config["system"]},
                    {"role": "user", "content": input}
                ]

            logger.debug(f"Chat request payload: {messages}")

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config["temperature"],
                    stream=self.stream,
                )

                async for part in response:
                    if not self.pause_stream:
                        content = part.choices[0].delta.content or ""
                        await self.output_buffer.put(content)

                if not self.pause_stream:
                    await self.output_buffer.put(None)
                    logger.info("Chat stream ended successfully")

            except Exception as e:
                logger.error(f"Error during chat stream: {e}")

    async def _describe_image(
        self, image: str | None, prompt: str = "Describe"
    ) -> str | None:
        """
        Describes an image using the vision model.
        """
        async with LLMClient._global_lock:
            logger.info(f"{self.mode.name} describing image")

            if not image:
                logger.warning("No image provided")
                return "No image provided"

            if self.mode == Mode.VISION:
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                                },
                            ],
                        }
                    ]
                    temp_client = openai.AsyncOpenAI(
                        base_url= "http://localhost:1313" + "/v1",
                        api_key="sk-no-key-required"
                    )
                    response = await temp_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.config.get("temperature", 0.7),
                    )
                    logger.debug(f"Image description response: {response}")
                    message_data = response.choices[0].message.content

                    if message_data:
                        return message_data
                    else:
                        logger.warning("No message found in response")
                        return "No message in response"

                except Exception as e:
                    logger.error(f"Error while describing image: {e}")
                    return "Error processing image"

    async def _fetch_response(self, input: str) -> str:
        """
        Fetches a complete response from a chat-style model.
        Compatible with Ollama / llama2 models.
        """
        try:
            # Prepare messages for chat
            messages = [
                {"role": "system", "content": self.config.get("system", "")},
                {"role": "user", "content": input}
            ]

            logger.info(f"Fetching response from model {self.model}")

            # Call chat completion API (async)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.get("temperature", 0.7),
                stream=False  # Ensure no streaming, we want a single complete response
            )

            # Defensive logging
            logger.debug(f"Full response object: {response!r}")

            # Extract the message content
            message_data = getattr(response.choices[0].message, "content", None)

            if not message_data:
                logger.warning("No message content found in response")
                return "No message in response"

            logger.info("Response received successfully")
            return message_data

        except Exception as e:
            logger.error(f"Error fetching response: {e}", exc_info=True)
            return "Error fetching response"

    async def _call_function(self, input: str, functions: list = []) -> Sequence | None:
        """
        Fetches a complete response from the model.
        """
        async with LLMClient._global_lock:
            logger.info(f"{self.mode.name} is fetching response")
            logger.info(f"Available tools: {[f.get('function', {}).get('name', '<unnamed>') for f in functions]}")

            try:
                messages = [
                    {"role": "system", "content": self.config["system"]},
                    {"role": "user", "content": input}
                ]
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=functions
                )

                # ─── Defensive logging & checks ──────────────────────────────────────
                if response is None:
                    logger.error("Client.chat.completions.create() returned None")
                    return None

                if not response.choices or not response.choices[0].message:
                    logger.error("No choices or message in response")
                    logger.debug(f"Full response object: {response!r}")
                    return None

                logger.debug(f"response.choices[0].message.content = {response.choices[0].message.content}")

                tool_calls = response.choices[0].message.tool_calls

                if tool_calls is None:
                    logger.info("tool_calls is explicitly None → treating as no tool call")
                    return None

                if tool_calls:  # now safe
                    logger.info("═══════════════════════════════════════════════")
                    logger.info("LLAMA.CPP CALLED TOOL(S):")
                    for i, tc in enumerate(tool_calls, 1):
                        name = tc.function.name
                        args = tc.function.arguments
                        logger.info(f"  ┌─ Tool #{i}")
                        logger.info(f"  │  name      : {name}")
                        logger.info(f"  │  arguments : {args}")
                        logger.info("  └──────────────────────────────────────")
                    return tool_calls
                else:
                    logger.info("No tool calls were made")
                    return None

            except Exception as e:
                logger.error(f"Error in _call_function: {e}", exc_info=True)
                return None   # ← better than returning string "Error…"

    @staticmethod
    async def fetch_embedding(text: str) -> np.ndarray | None:
        """
        Asynchronously fetches and caches an embedding for the given text while ensuring
        that no other locked operation (such as streaming) runs concurrently.
        """
        async with LLMClient._global_lock:
            client = openai.OpenAI(base_url=DEFAULT_HOST + "/v1", api_key="sk-no-key-required")
            try:
                logger.info("Fetching embedding")
                # Offload blocking work to a thread if needed
                response = await asyncio.to_thread(
                    client.embeddings.create, model=EMBEDDING_MODEL, input=text
                )
                embedding = response.data[0].embedding
                logger.debug(f"Extracted {len(embedding)} embeddings")
                return embedding
            except Exception as e:
                logger.error(
                    f"Error fetching embedding for text: {text}. Error: {str(e)}"
                )
                return