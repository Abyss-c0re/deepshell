import sys
import asyncio
from typing import Optional
from utils.logger import Logger

logger = Logger.get_logger()


class PipeUtils:
    def __init__(self, chat_manager):
        self.chat_manager = chat_manager
        self.processor = chat_manager.command_processor

    async def read_pipe(self):
        """
        Read piped input asynchronously.
        """
        loop = asyncio.get_event_loop()
        logger.info("Got the pipe content")
        return await loop.run_in_executor(None, sys.stdin.read)

    async def handle_pipe(self, user_input: Optional[str] = None) -> None:
        """
        Handles pipe input, formats the user input, and runs the task manager.
        """
        pipe_input = await self.read_pipe()

        if pipe_input:
            if user_input:
                user_input = self.processor.format_input(user_input, pipe_input)
            else:
                user_input = pipe_input
        results = await self.chat_manager.task_manager(user_input)
        print(results)
