import re
import shlex
import string
import asyncio
import secrets
from src.ui.printer import printer
from src.utils.logger import Logger
from src.config.settings import (
    SHELL_TYPE,
    MONITOR_INTERVAL,
    MAX_OUTPUT_LINES,
    FINALIZE_OUTPUT,
)

logger = Logger.get_logger()


class CommandExecutor:
    """
    A class for executing shell commands asynchronously, handling sudo authentication,
    monitoring execution, and processing command outputs.
    """

    def __init__(self, ui=None):
        """
        Initializes the CommandExecutor.

        Args:
            ui: Optional user interface object for interactive input/output.
            monitor_interval (int): Interval (in seconds) to check for long-running commands.
            max_output_length (int): Maximum length of output before truncation.
        """
        self.history = []
        self.ui = ui
        self._should_stop = False
        self.sudo_password = None
        self.monitor_interval = MONITOR_INTERVAL
        self.max_output_lines = MAX_OUTPUT_LINES
        self.finalize_output = FINALIZE_OUTPUT
        self.shell_type = SHELL_TYPE
        self.process: asyncio.subprocess.Process | None = None

        self.sudo_password = False

    async def start_shell(self) -> None:
        """
        Starts a persistent shell session if not already running.
        """

        if self.process is None or self.process.returncode is not None:
            self.process = await asyncio.create_subprocess_exec(
                self.shell_type,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            logger.info("Started persistent shell session.")

    async def run_command(self, command: str) -> str | None:
        """
        Executing command in a current shell session
        """
        if (
            self.process is None
            or self.process.stdin is None
            or self.process.stdout is None
        ):
            await self.start_shell()
            if (
                self.process is None
                or self.process.stdin is None
                or self.process.stdout is None
            ):
                logger.error("Failed to start shell process.")
                return "Error: Shell process could not be started."

        if "sudo" in shlex.split(command):
            valid = await self._get_sudo_password()
            if valid == 1:
                return None

        # Unique delimiter to detect command completion
        delimiter = "----END-OF-COMMAND----"
        full_command = f"{command} ; echo '{delimiter}' ; echo $?\n"

        logger.debug(f"Executing: {full_command}")
        self.process.stdin.write(full_command.encode())
        await self.process.stdin.drain()

        output_lines = []
        exit_code = None

        while True:
            line = await self.process.stdout.readline()
            if not line:
                break

            decoded_line = line.decode(errors="ignore").strip()

            if decoded_line == delimiter:
                # Read the exit code after delimiter
                exit_code_line = await self.process.stdout.readline()
                exit_code = int(exit_code_line.decode(errors="ignore").strip())
                break

            # Skip empty lines
            if decoded_line:
                output_lines.append(decoded_line)

        result = "\n".join(output_lines).strip()

        logger.info(f"Command produced {len(output_lines)} lines")
        logger.debug(f"Command result : {result}")

        if exit_code == 0 and not result:
            return "pass"

        if result and self.finalize_output:
            result = await self._finalize_command_output(output_lines)
            return result if result else f"Command failed with exit code {exit_code}"

    async def stop_shell(self):
        """
        Stops the persistent shell session.
        """
        if self.process and self.process.returncode is None:
            self.process.terminate()
            await self.process.wait()
            self.process = None
            logger.info("Shell session terminated.")
        else:
            logger.warning("Process already terminated or not started.")

    async def start(
        self, command: str
    ) -> tuple[str | None, str | None] | tuple[None, None]:
        """
        Starts execution of the given command.

        Args:
            command (str, optional): The command to execute.

        Returns:
            tuple: (confirmed_command, command output or error message)
        """
        logger.info("Execution started.")

        confirmed_command = None
        output = "No command specified."

        if command:
            logger.info("Command received, confirming execution.")
            confirmed_command = await self.confirm_execute_command(command)
            if confirmed_command:
                logger.info("Command confirmed, executing.")
                output = await self.run_command(confirmed_command)
            else:
                return None, None

        logger.info("Execution finished.")
        logger.debug(f"Command: {confirmed_command} Output: {output}")
        return confirmed_command, output

    async def execute_command(self, command: str) -> str | None:
        """
        Executes a shell command asynchronously without starting a shell session.

        Args:
            command (str): The command to execute.

        Returns:
            str: The command output or an error message.
        """
        if not command:
            return "No command provided."

        logger.info("Executing command.")

        require_sudo = True if "sudo" in shlex.split(command) else False

        if require_sudo:
            logger.info("Requesting sudo password.")
            sudo_password = await self._get_sudo_password(return_password=True)
            if sudo_password == 1:
                logger.warning("Sudo password not provided.")
                return None
            command = f"echo {sudo_password} | sudo -S {command[5:]}"
            sudo_password = None

        proc = await self._start_subprocess(command)
        if proc is None:
            logger.error("Subprocess creation failed.")
            return "Error: Command did not produce valid output or is not interactive."

        logger.info("Processing command output.")
        return await self._process_command_output(proc)

    async def _start_subprocess(self, command: str) -> asyncio.subprocess.Process:
        """
        Starts an asynchronous subprocess to execute a command.

        Args:
            command (str): The command to execute.

        Returns:
            subprocess.Process: The subprocess object.
        """
        return await asyncio.create_subprocess_shell(
            f"{self.shell_type} -c '{command}'",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _process_command_output(
        self, proc: asyncio.subprocess.Process
    ) -> str | None:
        """
        Processes the output of a running command.

        Args:
            proc (subprocess.Process): The running subprocess.
        Returns:
            str: The processed command output.
        """
        if not proc.stdout:
            logger.error("Shell subprocess is not running")
            return

        logger.info("Processing command output.")
        output_lines = []
        monitor_task = asyncio.create_task(self._monitor_execution(proc))

        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded_line = line.decode("utf-8", errors="ignore").strip()

                # Extract meaningful text
                extracted_text = self._extract_meaningful_text(decoded_line)
                if extracted_text:
                    output_lines.append(extracted_text)

                # Handle potential user prompts
                if self._should_handle_prompt(decoded_line):
                    await self._handle_prompt(proc, decoded_line)

        except asyncio.CancelledError:
            logger.warning("Command execution cancelled.")
            proc.terminate()
            await proc.wait()
            return "\n".join(output_lines)
        finally:
            monitor_task.cancel()

        # Validate if we received output
        if not output_lines:
            output_lines.append(
                "No output received. Command may require user interaction or is piped."
            )

        logger.info("Command output processing completed.")
        return await self._finalize_command_output(output_lines)

    def _extract_meaningful_text(self, data: str) -> str | None:
        """
        Cleans and extracts meaningful text from command output.

        Args:
            data (str): The raw command output.

        Returns:
            str: The cleaned output or None if empty.
        """
        # Strip out known non-text patterns (control codes, escape sequences)
        cleaned_data = re.sub(r"\x1b[^m]*m", "", data)  # Remove ANSI escape sequences
        cleaned_data = re.sub(
            r"[\x00-\x1F\x7F]", "", cleaned_data
        )  # Remove control characters

        # Try to extract the core content from the output
        if len(cleaned_data.strip()) > 0:
            return cleaned_data.strip()
        else:
            return

    async def _monitor_execution(self, proc: asyncio.subprocess.Process) -> None:
        """
        Periodically checks the status of a running process and prompts the user
        to cancel execution if it exceeds the monitoring interval.

        Args:
            proc (asyncio.subprocess.Process): The process being monitored.
        """
        while True:
            await asyncio.sleep(self.monitor_interval)
            if proc.returncode is not None:
                break
            user_choice = await self._get_user_input(
                "\nCommand is taking longer than expected. Cancel execution? (y/n): "
            )
            if user_choice.strip().lower() in ["y", "yes"]:
                printer("Terminating command execution...", True)
                proc.terminate()
                break

    async def _finalize_command_output(self, output_lines: list) -> str:
        """
        Finalizes the command output, ensuring truncation if too long and handling errors.

        Args:
            output_lines (list): Collected output lines.

        Returns:
            str: The final output.
        """
        logger.info("Finalizing command output.")

        # Truncate based on the number of lines
        if len(output_lines) > self.max_output_lines:
            logger.warning("Output truncated due to line limit.")
            output_lines = output_lines[: self.max_output_lines] + [
                "[Output truncated]"
            ]

        output_str = "\n".join(output_lines)

        logger.info("Command execution completed.")

        return output_str

    async def _get_sudo_password(self, return_password=False) -> str | int | None:
        """
        Prompts the user for a sudo password if not already provided.
        If the password was previously validated, revalidate sudo without a password.
        """
        if self.sudo_password:
            if await self._validate_sudo_password(None):
                return 0
            else:
                self.sudo_password = None
                logger.warning("Sudo session expired, password required again.")

        # Prompt for a new password
        sudo_password = await self._get_user_input(
            "Enter sudo password: ", is_password=True
        )
        if sudo_password:
            valid = await self._validate_sudo_password(sudo_password)
            if valid:
                if return_password:
                    return sudo_password
                self.sudo_password = True
                sudo_password = None
                logger.info("Sudo password validated and cleared securely.")
                return 0
            else:
                printer("Wrong password", True)
                logger.warning("Wrong sudo password")
                return 1

    async def _validate_sudo_password(self, sudo_password: str | None) -> bool:
        """
        Validates the provided sudo password or checks if sudo is still active.

        Args:
            sudo_password (str or None): The sudo password to validate, or None to check active status.

        Returns:
            bool: True if valid, False otherwise.
        """
        if sudo_password:
            command = f"echo {sudo_password} | sudo -S -v"
        else:
            # Check if sudo session is still active
            command = "sudo -n true"

        if self.process and self.process.stdin and self.process.stdout:
            full_command = f"stdbuf -oL {command}\n echo $?\n"
            self.process.stdin.write(full_command.encode())
            await self.process.stdin.drain()

            exit_code = None
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                decoded_line = line.decode(errors="ignore").strip()
                if decoded_line.isdigit():
                    exit_code = int(decoded_line)
                    break
            proc_valid = exit_code == 0
        else:
            proc = await self._start_subprocess(command)
            await proc.communicate()
            proc_valid = proc.returncode == 0

        if proc_valid:
            logger.info("Sudo session validated.")
            return True
        else:
            logger.warning("Sudo validation failed.")
            return False

    def _is_text(self, data: str) -> bool:
        """
        Determines whether the given data is likely to be human-readable text.

        Args:
            data (str): The output data to check.

        Returns:
            bool: True if the data is mostly printable text, False otherwise.
        """
        if not data or "\x00" in data:
            return False

        # Remove ANSI escape sequences
        data = re.sub(r"\x1b[^m]*m", "", data)

        # Calculate printable ratio
        printable_chars = sum(c in string.printable for c in data)
        printable_ratio = printable_chars / len(data) if len(data) > 0 else 0

        # Allow outputs with more than 70% printable characters (but filter out pure control codes or binary data)
        if printable_ratio < 0.7 or any(c in "\x07\x1b" for c in data):
            return False

        return True

    def _should_handle_prompt(self, decoded_line: str) -> bool:
        """
        Checks if a command output line contains a user prompt requiring a response.

        Args:
            decoded_line (str): The command output line.

        Returns:
            bool: True if the line contains a recognized prompt, False otherwise.
        """
        return any(
            kw in decoded_line.lower()
            for kw in ["[y/n]", "(yes/no)", "(y/n)", "password:", "continue?"]
        )

    async def _handle_prompt(
        self, proc: asyncio.subprocess.Process, decoded_line: str
    ) -> None:
        """
        Detects and responds to command-line prompts automatically.

        Args:
            proc (asyncio.subprocess.Process): The process awaiting input.
            decoded_line (str): The prompt message from the command output.
        """
        if not proc.stdin:
            logger.error("Shell subprocess is not running")
            return
        if "password:" in decoded_line.lower():
            response = await self._get_user_input("Enter password: ", is_password=True)
        else:
            response = "yes"

        if response is not None:
            proc.stdin.write(response.encode() + b"\n")
            await proc.stdin.drain()
            response = secrets.token_urlsafe(32)
        response = None

    async def confirm_execute_command(self, command: str) -> str | None:
        """
        Prompts the user to confirm and possibly edit a command before execution.

        Args:
            command (str): The command to confirm.

        Returns:
            str: The confirmed command or None if canceled.
        """
        command = command.lstrip()

        command = await self._get_user_input(
            "Validate the command and press [blue]Enter[/].\nOr delete the text and press [blue]Enter[/] to cancel",
            input_text=command,
        )

        return command if command else None

    async def _get_user_input(
        self,
        prompt_text: str = "Enter input: ",
        is_password: bool = False,
        input_text: str = "",
    ) -> str:
        """
        Helper function for retriving the user input, if UI is not loaded it will fallback to normal input
        """
        if self.ui is not None:
            return await self.ui.get_user_input(
                prompt_text, is_password=is_password, input_text=input_text
            )
        else:
            return input(prompt_text)
