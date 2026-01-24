import sys
import asyncio
from ui.ui import ChatMode
from ui.printer import printer
from utils.logger import Logger
from prompts.helper import PromptHelper
from chatbot.history import HistoryManager
from typing import Optional, Any, Callable
from chatbot.deployer import deploy_chatbot
from config.settings import Mode, PROCESS_IMAGES
from utils.command_processor import CommandProcessor
from core.task_manager import TaskManager


logger = Logger.get_logger()


class ChatManager:
    """
    Orchestrates chatbot execution, UI interaction, command handling,
    and delegates heavy async work to TaskManager.
    """

    def __init__(self):
        self.client, self.filtering = deploy_chatbot()

        self.ui = ChatMode(self) if self.client.render_output else None
        if not self.ui:
            self.client.keep_history = False

        self.last_mode: Optional[Mode] = None

        self.command_processor = CommandProcessor(self)
        self.file_utils = self.command_processor.file_utils
        self.executor = self.command_processor.executor

        self.tasks: list[asyncio.Task] = []

        self.task_executor = TaskManager()

    async def init(self):
        """Initialize history, UI helpers, and background services."""
        self.history_manager = HistoryManager(self)
        self.add_to_history = self.history_manager.add_message
        self.add_terminal_output = self.history_manager.add_terminal_output
        self.generate_prompt = self.history_manager.generate_prompt

        self.file_utils.set_index_functions(
            self.history_manager.add_file,
            self.history_manager.add_folder_structure,
        )

        await self.task_executor.start()
        await self.executor.start_shell()

    async def stop(self):
        """Gracefully stop background services."""
        await self.task_executor.stop()
        await self.executor.stop_shell()

    # ------------------------------------------------------------------
    # Task delegation
    # ------------------------------------------------------------------
    async def deploy_task(
        self,
        user_input: str,
        file_name: Optional[str] = None,
        file_content: Optional[str] = None,
    ) -> str | None:
        """
        Deploys a task based on user input and optional file input.
        """
        logger.info("Deploy task started.")
        response: Optional[str] = None
        action: Optional[str] = None

        if self.client.mode != Mode.VISION:
            self.last_mode = self.client.mode

        # --------------------------------------------------
        # File / pipe input handling
        # --------------------------------------------------

        if file_name:
            logger.info("Processing file: %s", file_name)
            await self.file_utils.process_file_or_folder(file_name)
            if not user_input:
                user_input = "Analyze this content"

        elif file_content:
            logger.info("Pipe input detected.")
            if not user_input:
                user_input = f"Analyze this: {file_content}"
            else:
                user_input = f"{user_input} Content: {file_content}"

        else:
            logger.info("Processing user input.")
            if self.client.mode != Mode.SHELL:
                processed_input, action = await self.command_processor.handle_command(
                    user_input
                )
                if processed_input:
                    user_input = processed_input

        # --------------------------------------------------
        # Task execution
        # --------------------------------------------------

        logger.info("Executing task manager.")

        if action or self.client.mode != Mode.DEFAULT:
            response = await self.task_manager(
                user_input=user_input,
                action=action,
            )
            if not response:
                logger.info("No response detected")
                return None

        if not sys.stdout.isatty():
            logger.info("Non-interactive stdout detected")
            return await self.task_manager(user_input=user_input)

        if self.client.keep_history and self.client.mode != Mode.SHELL and not response:
            history = await self.generate_prompt(user_input)
            response = await self.task_manager(history=history)

        # --------------------------------------------------
        # History & mode restoration
        # --------------------------------------------------

        if self.client.keep_history and response:
            await self.add_to_history("assistant", response)

        if self.last_mode and self.client.mode != self.last_mode:
            self.client.switch_mode(self.last_mode)

        logger.info("Deploy task completed.")
        return response

    async def deploy_chatbot_method(
        self,
        coro_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Backward-compatible proxy for enqueuing heavy async chatbot work.
        """
        return await self.task_executor.enqueue(coro_func, *args, **kwargs)

    # ------------------------------------------------------------------
    # Task routing
    # ------------------------------------------------------------------

    async def task_manager(
        self,
        user_input: str = "",
        history: Optional[list] = None,
        action: Optional[str] = None,
    ) -> str | None:
        """
        Routes tasks based on the current client mode.
        """
        if not user_input and not history and not action:
            logger.warning("Task manager invoked with no arguments")

        logger.info("Task manager started in mode: %s", self.client.mode)

        shell_bypass = action == "shell_bypass"
        action = action or ""

        mode_handlers = {
            Mode.SHELL: lambda inp: self._handle_shell_mode(inp, shell_bypass),
            Mode.CODE: self._handle_code_mode,
            Mode.VISION: lambda inp: self._handle_vision_mode(action, inp),
        }

        if shell_bypass:
            return await self._handle_shell_mode(user_input, True)

        handler = mode_handlers.get(self.client.mode)
        if handler:
            return await handler(user_input)

        return await self._handle_default_mode(input=user_input, history=history)

    # ------------------------------------------------------------------
    # Mode handlers
    # ------------------------------------------------------------------

    async def _handle_command_processor(self, input: str, functions: list):
        if self.client.mode != Mode.SYSTEM:
            self.client.switch_mode(Mode.SYSTEM)

        tools = await self.deploy_chatbot_method(
            self.client._call_function, input, functions
        )

        if self.last_mode:
            self.client.switch_mode(self.last_mode)

        return tools

    async def _handle_helper_mode(self, input: str, strip_json: bool = False) -> str:
        if self.client.mode != Mode.HELPER:
            self.client.switch_mode(Mode.HELPER)

        response = await self.deploy_chatbot_method(self.client._fetch_response, input)

        if strip_json:
            response = response.strip("`").strip("json")

        filtered = await self.deploy_chatbot_method(
            self.filtering.process_static, response
        )

        if self.last_mode:
            self.client.switch_mode(self.last_mode)

        return filtered

    async def _handle_vision_mode(
        self,
        target: str,
        user_input: str,
        no_render: bool = False,
    ) -> str | None:
        if not PROCESS_IMAGES:
            printer("Image processing is disabled (settings.py)", True)
            return None

        if self.client.mode != Mode.VISION:
            self.client.switch_mode(Mode.VISION)

        encoded_image = await self.file_utils._process_image(target)
        description = await self.deploy_chatbot_method(
            self.client._describe_image,
            image=encoded_image,
            prompt=user_input,
        )

        if self.last_mode:
            self.client.switch_mode(self.last_mode)

        if not no_render:
            printer(f"[green]AI:[/]{description}")

        return f"Image description by the vision model: {description}"

    async def _handle_shell_mode(
        self,
        input: str = "",
        bypass: bool = False,
        no_render: bool = False,
    ) -> str | None:
        logger.info("Shell mode execution started (bypass=%s)", bypass)

        if not bypass:
            code_input = await self._handle_code_mode(
                PromptHelper.shell_helper(input),
                shell=True,
            )
            command, output = await self.executor.start(code_input)
            if command:
                input = command
        else:
            self.client.switch_mode(Mode.SHELL)
            output = await self.executor.run_command(input)

        if output == "pass":
            printer("Command executed successfully with no output", True)
            return "pass"

        if not output or not input:
            printer("No output detected...", True)
            return None

        if self.ui:
            printer(f"Executing [green]'{input}'[/]", True)

            if await self.ui.yes_no_prompt(
                "Do you want to see the output?", default="No"
            ):
                self.tasks.append(
                    asyncio.create_task(
                        self.ui.fancy_print(f"[blue]Shell output[/]:\n{output}")
                    )
                )

            if await self.ui.yes_no_prompt("Analyze the output?", default="Yes"):
                await self.execute_tasks()
                prompt = PromptHelper.analyzer_helper(input, output)
                await self._handle_default_mode(input=prompt, no_render=no_render)

                if self.client.keep_history and self.client.last_response:
                    await self.add_terminal_output(
                        input,
                        output,
                        self.client.last_response,
                    )

                return self.client.last_response

        if self.client.keep_history:
            await self.add_terminal_output(input, output, "")

        return output

    async def _handle_code_mode(
        self,
        input: str,
        shell: bool = False,
        no_render: bool = False,
    ) -> str:
        response = await self.deploy_chatbot_method(self.client._fetch_response, input)

        if shell:
            return await self.filtering.extract_shell_command(response)

        code = await self.filtering.process_static(response, True)
        if code and not no_render:
            printer(code)

        return str(code)

    async def _handle_default_mode(
        self,
        input: Optional[str] = None,
        history: Optional[list] = None,
        no_render: bool = False,
    ) -> str | None:
        if self.ui and not no_render:
            if history and not input:
                chat_task = self.client._chat_stream(history=history)
            elif input and not history:
                chat_task = self.client._chat_stream(input)
            else:
                return None

            filter_task = self.filtering.process_stream(False, render=True)

            async def streaming_job():
                await asyncio.gather(chat_task, filter_task)

            await self.deploy_chatbot_method(streaming_job)
            return self.client.last_response

        response = await self.deploy_chatbot_method(self.client._fetch_response, input)
        return await self.filtering.process_static(response, False)

    async def execute_tasks(self) -> None:
        if not self.tasks:
            return
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
