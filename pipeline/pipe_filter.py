import re
from ui.printer import printer
from utils.logger import Logger
from ollama_client.api_client import OllamaClient

logger = Logger.get_logger()


class PipeFilter:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.input_buffer = ollama_client.output_buffer
        self.formatting = ollama_client.render_output
        self.extracted_code = None

    async def process_stream(
        self, extract_code: bool = False, render: bool = True
    ) -> None:
        """
        Processes the input stream, handling thoughts and code differently based on config.
        """
        full_input = ""
        results = ""

        if extract_code:
            while True:
                message = await self.input_buffer.get()
                if message is None:
                    break
                full_input += message

            self.ollama_client.last_response = full_input
            self.extracted_code = await self.extract_code(response=full_input)
            logger.debug(f"Extracted code: {self.extracted_code}")
            return

        # --- Default behavior: Process thoughts and full response ---
        thinking = False
        thought_buffer = []
        accumulated_line = ""
        first_chunk = True

        while True:
            chunk = await self.input_buffer.get()
            if chunk is None:
                break

            output = ""
            i = 0

            while i < len(chunk):
                if chunk[i:].startswith("<think>"):
                    thinking = True
                    i += 7
                    continue
                elif chunk[i:].startswith("</think>"):
                    thinking = False
                    i += 8
                    if self.ollama_client.show_thinking:
                        output += "\n[blue]Final answer:[/] "
                    continue

                char = chunk[i]

                if thinking:
                    thought_buffer.append(char)
                    if self.ollama_client.show_thinking:
                        output += char
                else:
                    output += char

                i += 1

            if output:
                if first_chunk:
                    prefix = "[purple]AI: [/]"
                    formatted_output = prefix + output.lstrip("\n")
                    accumulated_line += formatted_output
                    first_chunk = False
                else:
                    accumulated_line += output
                results += output

            if "\n" in accumulated_line:
                lines = accumulated_line.split("\n")
                for line in lines[:-1]:
                    if line.strip() and render:
                        printer(line)
                accumulated_line = lines[-1]

        if accumulated_line.strip() and render:
            printer(accumulated_line)

        self.ollama_client.last_response = results
        self.ollama_client.thoughts = thought_buffer
        logger.debug(f"PipeFilter output: {results} \nThoughts: {thought_buffer}")

    async def process_static(self, text: str, extract_code: bool = False) -> str:
        """
        Processes a static string, handling thoughts and code differently based on config.
        """
        if extract_code:
            self.ollama_client.last_response = text
            self.extracted_code = await self.extract_code(response=text)

            if self.extracted_code:
                logger.debug(f"Extracted code: {self.extracted_code}")
                return self.extracted_code

        # Process thoughts and filter them
        thoughts = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        filtered_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Apply additional line-based filtering
        pattern = re.compile(r"(#{3,4}|\*\*)")
        filtered_lines = [pattern.sub("", line) for line in filtered_text.splitlines()]
        filtered_text = "\n".join(filtered_lines)

        self.ollama_client.last_response = filtered_text
        self.ollama_client.thoughts.append(thoughts)

        logger.debug(f"Filtered text: {filtered_text} \nThoughts: {thoughts}")
        return filtered_text

    async def extract_shell_command(self, response: str) -> str:
        """
        Extracts a shell command from the response.
        """
        shell_pattern = r"```(?:sh|bash)\n(.*?)```"
        match = re.search(shell_pattern, response, re.DOTALL)

        if match:
            full_command = match.group(1).strip()
            commands = [
                line.strip() for line in full_command.split("\n") if line.strip()
            ]
            command = commands[0] if len(commands) == 1 else " && ".join(commands)
        else:
            single_line = response.strip().split("\n")
            command = (
                single_line[0].strip()
                if len(single_line) == 1 and single_line[0]
                else ""
            )

        command = command.replace("\n", " && ").strip("`").strip()
        return command

    async def extract_code(self, response: str) -> str | None:
        """
        Extracts all code snippets from the response and separates them with comments if needed.
        """
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        code_snippets = []
        for i, match in enumerate(matches):
            language, code = match
            code = code.strip()
            if self.formatting:
                if i > 0:
                    code_snippets.append("\n# --- Next Code Block ---\n")
                code_snippets.append(f"```{language}\n{code}```")
            else:
                if i > 0:
                    code_snippets.append("\n# --- Next Code Block ---\n")
                code_snippets.append(code)

        return "\n".join(code_snippets) if code_snippets else None
