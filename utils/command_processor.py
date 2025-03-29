import os
import re
from ui.printer import printer
from utils.logger import Logger
from config.settings import Mode
from typing import Optional, Tuple
from utils.file_utils import FileUtils
from utils.shell_utils import CommandExecutor

logger = Logger.get_logger()


class CommandProcessor:
    """
    Handles user input
    """
    
    def __init__(
            self, 
            manager 
    ):
        self.manager = manager
        self.ui = manager.ui
        self.file_utils = FileUtils(manager)
        self.executor = CommandExecutor(self.ui)


    async def handle_command(
            self, 
            user_input: str
    )-> Tuple[str,str | None] | Tuple[None,None]:
        """
        Processes commands, handles file/folder operations, and updates config.
        """
        
        # Handle shell bypass if user input starts with "!"
        if user_input.startswith("!"):
            return user_input[1:], "shell_bypass"

        # Handle @mode command
        if user_input.startswith("@"):
            input = await self.detect_mode(user_input)
            if input is None:
                return "", None
            else:
                if self.manager.client.mode == Mode.SHELL:
                    return input, None
        # If there is a user input to process
        if user_input:
            target, additional_action = await self.detect_action(user_input)
            
            if target:
                pass_image = False
                
                # Check if the target is an image, switch to vision mode if true
                if self.file_utils._is_image(target):
                    pass_image = True
                    self.manager.client.switch_mode(Mode.VISION)
                else:
                    await self.file_utils.process_file_or_folder(target)
                    
                # If there's an additional action to update the user input
                if additional_action:
                    user_input = additional_action
                    
                # Return the user input along with target if it's an image
                if pass_image:
                    return user_input, target
                
                return user_input, None
            else:
                # If no target was found and action is "cancel", return None
                if additional_action == "cancel":
                    return None, None

        # Return the original user input if no specific handling was done
        return user_input, None

   
    async def detect_mode(
            self, 
            user_input: str
    ) -> str | None:
        """
        Detects if input starts with @ and checks if it matches a Mode.
        """
        mode_switcher = self.manager.client.switch_mode
        parts = user_input.split(" ", 1)
        
        if len(parts) < 2:
            if self.ui:
                await self.ui.fancy_print("\n\n[red]System:[/]\nNo user prompt detected after mode override\n")
            logger.warning("No user prompt detected after user override")
            return None

        mode_str, after_text = parts[0][1:].upper(), parts[1]
        
        try:
            mode = Mode[mode_str]
            
            # Special handling for Mode.VISION
            if mode == Mode.VISION:
                logger.warning("Mode.VISION cannot be selected directly.")
                return None

            mode_switcher(mode)
            logger.info(f"Mode detected: {mode.name}")
            return after_text
            
        except KeyError:
            printer("Invalid mode override",True)
            logger.warning("Invalid mode override, suspending input")
            return None
        except Exception as e:
            printer(str(e),True)
            logger.warning(str(e))
            return None

 
    async def detect_action(
            self, 
            user_input: str
    ):
        """
        Detects action, validates/finds target, and processes file/folder.
        """

        parts = re.split(r"\band\b", user_input, maxsplit=1)
        main_command = parts[0].strip()
        additional_action = parts[1].strip() if len(parts) > 1 else None
        actions = {"find", "open", "read"}
        tokens = main_command.split()

        if not tokens:
            return None, None

        # Ensure the action is at the beginning of the input
        action = tokens[0] if tokens[0] in actions else None
        if not action:
            return None, None

        # Extract target (everything after the action)
        target_index = 1  # Start from the second token, assuming it's the target
        target = " ".join(tokens[target_index:]) if target_index < len(tokens) else ""

        # Convert "this folder" to current working directory
        if target.lower() == "this folder":
            target = os.getcwd()

        # Validate or find target
        target = target.strip()
        if not os.path.exists(target):
            choice = await self.file_utils.prompt_search(target)
            if not choice:
                printer("Nothing found",True)
                return None, None
            if choice == "cancel" or choice == "nothing":
                printer("Search canceled by user",True)
                return choice
            target = choice

        # Ensure default additional action if none is specified
        if target:
            if not additional_action:
                additional_action = f"Analyze the content of {target}"
            else:
                additional_action = f"{additional_action} for {target}"

        return target, additional_action

    def format_input(
            self, 
            user_input: str, 
            file_content: str, 
            additional_action:  Optional[str] = None
    ):
        """
        Prepares user input by combining prompt and file content.
        """
        formatted_content = f"Content:\n{file_content}"
        if additional_action:
            user_input = additional_action
        if user_input:
            return f"\n{formatted_content}\nUser Prompt:\n{user_input}\n"
        return formatted_content
