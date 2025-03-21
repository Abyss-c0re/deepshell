import time
import asyncio
from ui.ui import ChatMode
from utils.logger import Logger
from chatbot.helper import PromptHelper
from chatbot.history import HistoryManager
from chatbot.deployer import ChatBotDeployer
from config.settings import Mode, PROCESS_IMAGES
from utils.command_processor import CommandProcessor

logger = Logger.get_logger()

class ChatManager:
    """
    Manages the chatbot's operations, including initializing the client, handling user commands,
    managing tasks, and rendering outputs.
    """

    def __init__(self):
        self.client, self.filtering = ChatBotDeployer.deploy_chatbot()
        self.ui = ChatMode(self) if self.client.render_output else None
        self.last_mode = None

        self.command_processor = CommandProcessor(self)
        self.file_utils = self.command_processor.file_utils
        self.executor = self.command_processor.executor

        self.tasks = []
        self.task_queue = asyncio.Queue()
        self.worker_running = False

      
    async def init(self):
        """
        Helper function to initialize ChatMode.
        """
        self.history_manager = HistoryManager(self)
        self.add_to_history = self.history_manager.add_message
        self.add_terminal_output = self.history_manager.add_terminal_output
        self.generate_prompt = self.history_manager.generate_prompt
        self.file_utils.set_index_functions(
            self.history_manager.add_file,
            self.history_manager.add_folder_structure
        )

        self.worker_task = asyncio.create_task(self.task_worker())
        await self.executor.start_shell()

    async def stop(self):
        """
        Helper function to stop the ChatMode.
        """
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
                logger.info("Worker process is terminated")
            except asyncio.CancelledError:
                logger.error("Worker task cancelled") 
        await self.executor.stop_shell()

    async def deploy_task(self, user_input=None, file_name=None, file_content=None):
        """
        Deploys a task based on user input and file content.
        """
        logger.info("Deploy task started.")
        response = None
        if self.client.mode != Mode.VISION:
            self.last_mode = self.client.mode

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
            logger.info("No file content, processing user input.")
           #await self.command_processor.ai_handler(user_input)
            user_input = await self.command_processor.handle_command(user_input)

        user_input, bypass_flag = (
            user_input if isinstance(user_input, tuple) else (user_input, False)
        )

        logger.info("Executing task manager.")

        if bypass_flag or self.client.mode != Mode.DEFAULT:
            response = await self.task_manager(user_input=user_input, bypass=bypass_flag)
            if not response:
                logger.info("No response detected")
                return

        if self.client.keep_history and self.client.mode != Mode.SHELL and not response:
            history = await self.generate_prompt(user_input)
            response = await self.task_manager(history=history)

        if self.client.keep_history and response:
            await self.add_to_history("assistant", response)

        if self.client.mode != self.last_mode:
            self.client.switch_mode(self.last_mode)

        logger.info("Deploy task completed.")
        return response

    async def deploy_chatbot_method(self, coro_func, *args, **kwargs):
        """
        Enqueue a heavy chatbot processing call (e.g. _chat_stream, _fetch_response,
        process_static, _describe_image, etc.) and return its result.
        """
        future = asyncio.get_running_loop().create_future()
        await self.task_queue.put((coro_func, args, kwargs, future))
        if not self.worker_running:
            logger.info("Starting task worker from deploy_chatbot_method.")
            asyncio.create_task(self.task_worker())
        return await future

    async def task_worker(self):
        """
        Processes tasks from the unified queue sequentially.
        Distinguishes heavy processing calls (tuples of 4 elements) and logs execution times
        and queue sizes.
        """
        if self.worker_running:
            logger.info("Task worker already running.")
            return
        self.worker_running = True
        logger.info("Task worker started.")

        while not self.task_queue.empty():
            queue_size_before = self.task_queue.qsize()
            start_time = time.time()
            task = await self.task_queue.get()

            # Heavy chatbot processing call: (coro_func, args, kwargs, future)
            if len(task) == 4:
                coro_func, args, kwargs, future = task
                try:
                    result = await coro_func(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    logger.error("Chatbot task error: %s", e)
                    future.set_exception(e)
            else:
                # In case legacy tasks are ever added to this queue.
                logger.warning("Encountered legacy task in heavy task queue. Skipping.")

            end_time = time.time()
            execution_time = end_time - start_time
            queue_size_after = self.task_queue.qsize()
            logger.info("Task completed in %.2f seconds. Queue size before: %d, after: %d",
                        execution_time, queue_size_before, queue_size_after)
            self.task_queue.task_done()

        logger.info("No more tasks. Task worker is going idle.")
        self.worker_running = False

    async def task_manager(self, user_input=None, history=None, bypass=None):
        """
        Manages tasks based on the client's mode.
        """
        logger.info("Task manager started in mode: %s", self.client.mode)

        shell_bypass = True if bypass == "shell" else False
        if bypass is None:
            bypass = ""

        mode_handlers = {
            Mode.SHELL: lambda inp: self._handle_shell_mode(inp, shell_bypass),
            Mode.CODE: self._handle_code_mode,
            Mode.VISION: lambda inp: self._handle_vision_mode(bypass, inp),
        }

        if shell_bypass:
            logger.info("Bypassing mode, executing shell mode.")
            return await self._handle_shell_mode(user_input, True)

        if self.client.mode in mode_handlers:
            logger.info("Handling task in mode: %s", self.client.mode)
            return await mode_handlers[self.client.mode](user_input)
        else:
            logger.info("Handling task in default mode.")
            return await self._handle_default_mode(input=user_input, history=history)

    async def _handle_command_processor(self, input,functions):
        if self.client.mode != Mode.SYSTEM:
            self.client.switch_mode(Mode.SYSTEM)
                                 
        tools = await self.deploy_chatbot_method(self.client._call_function, input, functions)
       
        self.client.switch_mode(self.last_mode)

        return tools


    async def _handle_helper_mode(self, input,strip_json = False):
        if self.client.mode != Mode.HELPER:
            self.client.switch_mode(Mode.HELPER)

        response = await self.deploy_chatbot_method(self.client._fetch_response, input)
        if strip_json:
            response = response.strip("`").strip("json")

        filtered_response = await self.deploy_chatbot_method(self.filtering.process_static, response)

        if self.last_mode:
            self.client.switch_mode(self.last_mode)
        return filtered_response

    async def _handle_vision_mode(self, target, user_input, no_render=False):
        """
        Handles vision mode by generating an image description.
        The heavy call to describe the image is offloaded.
        """
        if PROCESS_IMAGES:
            if self.client.mode != Mode.VISION:
                self.client.switch_mode(Mode.VISION)
            logger.info("Processing image %s", target)
            encoded_image = await self.file_utils._process_image(target)
            description = await self.deploy_chatbot_method(
                self.client._describe_image, image=encoded_image, prompt=user_input
            )
            logger.info("Processed image %s", target)
            if self.last_mode:
                self.client.switch_mode(self.last_mode)
            if self.ui and not no_render:
                await self.ui.fancy_print(f"[green]AI:[/]{description}")
            return f"Image description by the vision model: {description}"
        else:
            if self.ui:
                await self.ui.fancy_print("[cyan]System:[/]Image processing is disabled, check your settings.py")
            logger.warning("Image processing is disabled")
            return None

    async def _handle_shell_mode(self, input, bypass=False, no_render=False):
        """
        Handles tasks when the client is in SHELL mode.
        Command execution is performed immediately, while heavy processing (code conversion
        and output analysis) is offloaded.
        """
        logger.info("Shell mode execution started. Bypass: %s", bypass)
        if not bypass:
            code_input = await self._handle_code_mode(PromptHelper.shell_helper(input), no_render=True)
            input, output = await self.executor.start(code_input)
        else:
            self.client.switch_mode(Mode.SHELL)

            output = await self.executor.run_command(input)

        if output == "pass":
            if self.ui:
                await self.ui.fancy_print("[cyan]System:[/] Command executed successfully")
            return "pass"

        if output and input:
            logger.info("Command executed, processing output.")
            if self.ui:
                await self.ui.fancy_print(f"[cyan]System:[/] Executing [green]'{input}'[/]")
                if await self.ui.yes_no_prompt("Do you want to see the output?", default="No"):
                    render_task = asyncio.create_task(self.ui.fancy_print(f"[blue]Shell output[/]:\n{output}"))
                    self.tasks.append(render_task)
                await asyncio.sleep(0.1)
                if await self.ui.yes_no_prompt("Analyze the output?", default="Yes"):
                    if self.tasks:
                        asyncio.create_task(self.execute_tasks())
                    await self.ui.fancy_print("[cyan]System:[/] Output submitted to the chatbot for analysis...")
                    prompt = PromptHelper.analyzer_helper(input, output) 
                    get_summary = self.deploy_chatbot_method(self.client._chat_stream, prompt)
                    filter_summary = self.deploy_chatbot_method(self.filtering.process_stream, False, render=not no_render)
                    await asyncio.gather(get_summary, filter_summary)
                    if self.client.keep_history and self.client.last_response:
                        await self.add_terminal_output(input, output, self.client.last_response)
                    return self.client.last_response
                else:
                    if self.tasks:
                        await asyncio.gather(*self.tasks)
                    if self.client.keep_history:
                        await self.add_terminal_output(input, output, "")
                    return output
        else:
            logger.warning("No output detected.")
            if self.ui:
                await self.ui.fancy_print("\nNo output detected...\n")

        self.client.last_response = ""
        self.filtering.extracted_code = ""
        logger.info("Shell mode execution completed.")
        return output

    async def _handle_code_mode(self, input, no_render=False):
        """
        Handles tasks when the client is in CODE mode.
        Heavy processing (fetching response and static processing) is offloaded.
        """
        logger.info("Code mode execution started.")
        response = await self.deploy_chatbot_method(self.client._fetch_response, input)
        code = await self.deploy_chatbot_method(self.filtering.process_static, response, True)
        if code:
            if self.ui and not no_render:
                await self.ui.fancy_print(code)
            return code

    async def _handle_default_mode(self, input=None, history=None, no_render=False):
        """
        Handles tasks when the client is in the default mode.
        Streaming and filtering are queued together as one job and passed to deploy_chatbot.
        """
        logger.info("Default mode execution started.")

        client = self.client
        filtering = self.filtering

        if history and not input:
            logger.info("Using chat history.")
            chat_task = client._chat_stream(history=history)
        elif input and not history:
            logger.info("Using user input.")
            chat_task = client._chat_stream(input)
        else:
            logger.error("Invalid input.")
            return

        # Decide whether to render output
        rendering = True if self.ui and not no_render else False
        filter_task = filtering.process_stream(False, render=rendering)

        # Create a single async job for both tasks
        async def streaming_job():
            await asyncio.gather(chat_task, filter_task)

        # Pass the job to deploy_chatbot_method
        await self.deploy_chatbot_method(streaming_job)
        return self.client.last_response


    async def execute_tasks(self):
        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except Exception as e:
            logger.error("Error in default mode execution: %s", e)
        self.tasks = []

