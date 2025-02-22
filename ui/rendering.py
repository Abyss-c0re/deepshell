import re
import asyncio
from utils.logger import Logger

logger = Logger.get_logger()

class Rendering:
    def __init__(self, chat_app):
        self.chat_app = chat_app

    async def render_output(self):
        """Rendering the content from the buffer,word by word"""
        accumulated_text = ""
        while True:
            chunk = await self.chat_app.buffer.get()
            if chunk is None:
                break
            chunk = chunk.replace("###", "")
            chunk = chunk.replace("####", "")
            chunk = chunk.replace("**", "")

            accumulated_text += chunk
            self.chat_app.rich_log_widget.clear()
            self.chat_app.rich_log_widget.write(accumulated_text)
            self.chat_app.rich_log_widget.scroll_end()
        await asyncio.sleep(0.01)


    async def fancy_print(self, content, delay=0.01):
        """Render string word by word, preserving newlines and other whitespace."""
        while not self.chat_app.buffer.empty():
            await asyncio.sleep(0.01) 
        chunks = re.split(r'(\s+)', content)
        
        for chunk in chunks:
            if chunk == '\n':  # Handle newline separately to ensure correct formatting
                await self.chat_app.buffer.put('\n')  # Send newline directly to the buffer
            else:
                await self.chat_app.buffer.put(chunk)  # Send words to buffer
            await asyncio.sleep(delay) 
    

    async def transfer_buffer(self, content):
        """
        Continuously transfer data from the source_buffer (e.g. filtering's buffer)
        into the UI's rendering buffer, but only if the transfer is enabled.
        """

        while not self.chat_app.buffer.empty():
            await asyncio.sleep(0.01)

        if isinstance(content, asyncio.Queue):
            while True:
                chunk = await content.get()
                if chunk is None:
                    break
                await self.chat_app.buffer.put(chunk)
        elif hasattr(content, "__aiter__"):
            async for chunk in content:
                await self.chat_app.buffer.put(chunk)
        else:
             await self.chat_app.buffer.put(content)

