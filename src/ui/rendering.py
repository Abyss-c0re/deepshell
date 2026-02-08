import re
import asyncio
from src.config.settings import RENDER_DELAY


class Rendering:
    _chat_app_instance = None

    def __init__(self, chat_app):
        self.chat_app = chat_app
        Rendering._chat_app_instance = chat_app
        self.cleaner = re.compile(r"(#{3,4}|\*\*)")
        self.delay = RENDER_DELAY
        self._lock = asyncio.Lock()
        self.queue = asyncio.Queue()
        self._processing_task = None  # Don't start in __init__, defer it

    async def start_processing(self) -> None:
        """
        Start queue processing task after event loop is running.
        """
        if not self._processing_task:
            self._processing_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self) -> None:
        """
        Continuously process print jobs from the queue.
        """
        while True:
            content = await self.queue.get()
            await self._execute_fancy_print(content)
            self.queue.task_done()

    async def _execute_fancy_print(
        self,
        content: str,
    ) -> None:
        """
        Render string line by line, preserving newlines and whitespace.
        """
        lines = content.split("\n")
        if len(lines) > 1:
            self.chat_app.lock_input()

        for line in lines:
            await self.render_output(line)
            await asyncio.sleep(self.delay)

        self.chat_app.unlock_input()

    async def render_output(self, line: str) -> None:
        """
        Render lines while stripping some markdown tags.
        """
        async with self._lock:
            cleaned_line = self.cleaner.sub("", line.rstrip())
            self.chat_app.rich_log_widget.write(cleaned_line)

    async def fancy_print(self, content: str) -> None:
        """
        Add print job to queue and ensure execution order.
        """
        await self.queue.put(content)

    @staticmethod
    async def _fancy_print(content: str) -> None:
        """
        Static method to enqueue print job.
        """
        if Rendering._chat_app_instance:
            await Rendering._chat_app_instance.rendering.fancy_print(content)
