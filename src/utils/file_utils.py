import os
import magic
import base64
import asyncio
import aiofiles
from PIL import Image
from io import BytesIO
from typing import Callable
from src.ui.printer import printer
from src.utils.logger import Logger

from src.ui.popups import RadiolistPopup
from src.config.settings import (
    IGNORE_DOT_FILES,
    SUPPORTED_EXTENSIONS,
    IGNORED_FOLDERS,
    MAX_FILE_SIZE,
    MAX_LINES,
    PROCESS_IMAGES,
    IMG_INPUT_RES,
)

logger = Logger.get_logger()


class FileUtils:
    def __init__(self, manager):
        self.ui = manager.ui
        self.index_file = None
        self.add_folder = None
        self.file_locks = {}

        if PROCESS_IMAGES:
            self.image_processor = manager._handle_vision_mode

    def set_index_functions(self, index_file: Callable, add_folder: Callable) -> None:
        """
        Helper function to avoid circular import
        """
        self.index_file = index_file
        self.add_folder = add_folder

    async def process_file_or_folder(self, target: str) -> str | int | None:
        """
        Processing the target, if exists.
        If not, tries to find the path.
        Returns full path if target is confirmed, esle -1
        """
        target = target.strip()

        if not os.path.exists(target):
            choice = await self.prompt_search(target)
            if not choice:
                printer("[yellow]Nothing found[/yellow]", True)
                return -1
            target = choice

        if os.path.isfile(target):
            content = await self.read_file(target)
            if self.index_file:
                await self.index_file(target, content)
        elif os.path.isdir(target):
            await self.read_folder(target)

        logger.info("File operations complete")
        printer("File processing complete, submiting the input to the chatbot", True)

    async def read_file(
        self,
        file_path: str,
        max_file_size: int = MAX_FILE_SIZE,
        max_lines: int = MAX_LINES,
    ) -> str | None:
        """
        Returns the content of a file or image description from the VISION model
        """
        try:
            if not self._is_safe_file(file_path):
                logger.info(f"Skipping file (unsupported): {file_path}")
                return None

            # Create a lock for this file if it doesn't exist yet
            if file_path not in self.file_locks:
                self.file_locks[file_path] = asyncio.Lock()

            # Use the lock for this file
            async with self.file_locks[file_path]:
                if PROCESS_IMAGES:
                    if self._is_image(file_path):
                        printer(f"Processing the image: {file_path}", True)
                        description = await self.image_processor(
                            file_path, "Describe this image", True
                        )
                        logger.info(f"Processed {file_path}")
                        return f"Image description by the vision model: {description}"

                else:
                    if self._is_image(file_path):
                        logger.info(f"Skipping image file {file_path}")

                printer(f"Reading {file_path}", True)
                content = await _read_file(file_path)
                if content:
                    n_lines = len(content.splitlines())
                    logger.info(f"Read {n_lines} lines of text")
                    if (
                        os.path.getsize(file_path) > max_file_size
                        or n_lines > max_lines
                    ):
                        content = await self._read_last_n_lines(content, max_lines)

                return content

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")

    def _is_safe_file(
        self, file_path: str, supported_extensions: list = SUPPORTED_EXTENSIONS
    ) -> bool:
        """
        Returns True if file located at provided file_path:str is in the SUPPORTED_EXTENSIONS list
        or if the file is without extension is a text file.
        """
        if os.path.getsize(file_path) == 0:
            logger.info(f"Skipping empty file: {file_path}")
            return False

        if any(file_path.lower().endswith(ext) for ext in supported_extensions):
            return True

        if "." not in os.path.basename(file_path):
            logger.info(
                f"File '{file_path}' has no extension. Checking if it's a text file..."
            )
            return self._is_text_file(file_path)

        return False

    def _is_text_file(self, file_path: str) -> bool:
        """
        Checks if provided file at the file_path:str is a text file
        Return bool
        """
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(file_path)

            logger.info(f"Detected MIME type for '{file_path}': {mime_type}")

            if mime_type.startswith("text"):
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error detecting MIME type for '{file_path}': {e}")
            return False

    def _is_image(self, file_path: str) -> bool:
        """Checks if file at provided file_path:str is an image
        Returns bool
        """
        try:
            mime = magic.Magic(mime=True)
            return mime.from_file(file_path).startswith("image")
        except Exception:
            return False

    async def _process_image(
        self,
        file_path: str,
        encoding_format: str = "PNG",
        img_size: tuple[int, int] = IMG_INPUT_RES,
    ) -> str | None:
        """
        Open image at provided file_path:str and encodes it as base64:str
        """
        logger.info(f"Encoding image: {file_path}")
        loop = asyncio.get_running_loop()
        try:

            def resize_and_encode():
                with Image.open(file_path) as img:
                    img.thumbnail(img_size)
                    buffer = BytesIO()
                    img.save(buffer, format=encoding_format)
                    return base64.b64encode(buffer.getvalue()).decode("utf-8")

            encoded_image = await loop.run_in_executor(None, resize_and_encode)
            return encoded_image
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")

    async def _read_last_n_lines(
        self, file_content: str, num_lines: int, last: bool = True
    ) -> str:
        """
        Function to return lines from file content.
        If 'last' is True, returns the last `num_lines` lines; otherwise, returns the first `num_lines`.
        """
        buffer = []
        lines = file_content.splitlines()

        if last:
            start_index = max(0, len(lines) - num_lines)
            buffer = lines[start_index:]
        else:
            end_index = min(len(lines), num_lines)
            buffer = lines[:end_index]

        return "\n".join(buffer) if buffer else "[File is empty]"

    def _get_file_size(self, file_path: str) -> int:
        """
        Returns the file size
        """
        try:
            with open(file_path, "rb") as f:
                f.seek(0, 2)
                return f.tell()
        except Exception:
            return 0

    def generate_structure(
        self,
        folder_path: str,
        root_folder: str,
        prefix: str = "",
        ignored_folders: list = IGNORED_FOLDERS,
        ignore_dot_files: bool = IGNORE_DOT_FILES,
    ) -> dict:
        """
        Generates a dictionary representation of the folder structure.
        All files within the folder are included, regardless of file type.
        """
        logger.info(f"Generating structure for {folder_path}")

        structure = {}
        folder_name = os.path.basename(folder_path)
        structure[folder_name] = {}

        try:
            items = sorted(os.listdir(folder_path))
        except Exception as e:
            logger.error(f"Error reading folder {folder_path}: {e}")
            return structure

        for item in items:
            item_path = os.path.join(folder_path, item)
            if (
                os.path.isdir(item_path)
                and item not in ignored_folders
                and (not ignore_dot_files and item.startswith("."))
            ):
                structure[folder_name][item] = self.generate_structure(
                    item_path, root_folder, prefix + "--"
                )
            elif os.path.isfile(item_path):
                relative_path = (
                    os.path.relpath(item_path, root_folder)
                    if root_folder
                    else item_path
                )
                structure[folder_name][item] = relative_path

        logger.debug(f"Folder structure: {structure}")
        return structure

    async def read_folder(
        self,
        folder_path: str,
        root_folder: str | None = None,
        ignored_folders: list = IGNORED_FOLDERS,
    ) -> str | None:
        """
        Recursively scans and reads all files in a folder.
        The folder structure is generated for all files; however, only files with safe extensions
        are attempted to be read (others are skipped).
        """
        logger.info(f"Opening {folder_path}")
        printer(f"Opening {folder_path}", True)

        if root_folder is None:
            root_folder = folder_path

        all_contents = []  # To collect content from all files

        try:
            printer(f"Generating structure for {folder_path}", True)
            generated_structure = self.generate_structure(folder_path, root_folder)
            if self.add_folder:
                self.add_folder(generated_structure)

            # Collecting content of all files
            for root, _, files in os.walk(folder_path):
                if any(ignored in root.split(os.sep) for ignored in ignored_folders):
                    continue
                for file in files:
                    file_path = os.path.join(root, file)

                    content = await self.read_file(file_path)
                    if self.index_file and content:
                        await self.index_file(file_path, content, folder=True)
                    elif content:
                        file_contents = f"\n{content.strip()}\n"
                        if file_contents:
                            all_contents.append(file_contents)

            logger.info(f"Reading files in {folder_path} complete")

            return "\n".join(all_contents)

        except PermissionError:
            logger.error(f"Error: Permission denied to access '{folder_path}'.")

    async def search_files(self, missing_path: str = "", search_dir=None) -> list:
        """
        Searches for a missing file or folder in the specified directory.
        Defaults to the home directory if none is provided.
        """
        if not search_dir:
            search_dir = os.path.expanduser("~")

        results = []
        for root, dirs, files in os.walk(search_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            files = [f for f in files if not f.startswith(".")]

            for name in files + dirs:
                if missing_path.lower() in name.lower():
                    results.append(os.path.join(root, name))
        logger.info(f"Found {len(results)} files")
        return results

    async def prompt_search(self, missing_path: str) -> str:
        """
        Prompts the user to search for a file when the target is missing.
        Uses a popup with a radiolist if UI is available, falling back to terminal input.
        """
        while True:
            results = await self.search_files(missing_path)

            if not results:
                return "nothing"

            options = [(res, res) for res in results] + [("cancel", "Cancel")]
            if hasattr(self, "ui") and self.ui is not None:
                popup = RadiolistPopup(
                    title="Select a file",
                    text=f"Multiple matches found for '{missing_path}'. Please choose one:",
                    options=options,
                )
                self.ui.mount(popup)
                choice = await popup.wait_for_choice()
                popup.remove()
            else:
                print(
                    f"Multiple matches found for '{missing_path}'. Please choose one:"
                )
                for i, res in enumerate(results, start=1):
                    print(f"{i}. {res}")
                choice_str = input(
                    "Enter the number of your choice (or 'cancel'): "
                ).strip()
                if choice_str.lower() == "cancel":
                    return "cancel"
                if choice_str.isdigit() and 1 <= int(choice_str) <= len(results):
                    choice = results[int(choice_str) - 1]
                else:
                    print("Invalid input, try again.")
                    continue

            if choice == "cancel":
                return "cancel"
            return choice


async def _read_file(file_path: str) -> str | None:
    """
    Asynchronously reads a file.

    Args:
        file_path (str): The file's path.

    Returns:
        str: The file's content.
    """
    try:
        async with aiofiles.open(
            file_path, "r", encoding="utf-8", errors="ignore"
        ) as f:
            content = await f.read()
        return content
    except (IOError, OSError) as e:
        logger.error(f" Error reading file {file_path}: {str(e)}")
        return None
