import os
import aiofiles
import asyncio
import magic
from ui.popups import RadiolistPopup
from config.file_utils_config import *

class FileUtils:
    def __init__(self, ui=None, safe_extensions=None, ignore_folders=None, scan_dot_folders=False):
        """
        Initializes the FileUtils with options.

        :param ui: A UI object (optional).
        :param safe_extensions: List of file extensions that are allowed.
               If not provided, a default whitelist is used.
        :param ignore_folders: List of folder names to ignore.
        :param scan_dot_folders: Whether to scan hidden folders (starting with a dot). Default is False.
        """
        self.ui = ui
        self.default_safe_extensions = SUPPORTED_EXTENSIONS
        self.safe_extensions = safe_extensions or self.default_safe_extensions
        self.default_ignore_folders = IGNORED_FOLDERS       
        self.ignore_folders = ignore_folders or self.default_ignore_folders
        self.scan_dot_folders = scan_dot_folders
        self.max_file_size = MAX_FILE_SIZE
        self.max_lines = MAX_LINES
        self.chunk_size = CHUNK_SIZE

    async def process_file_or_folder(self, target):
        """Handles file or folder operations.""" 
        target = target.strip()
       
        if not os.path.exists(target):
            choice = await self.prompt_search(target)
            if not choice:
                await self._print_message("\n[yellow]Nothing found[/yellow]")
                return -1
            target = choice

        if os.path.isfile(target):
            return await self.read_file(target)
        elif os.path.isdir(target):
            return await self.read_folder(target)

        return None

    async def read_file(self, file_path, root_folder=None):
        """Reads a file if it's safe or has no extension but matches known patterns."""
        try:
            if not self._is_safe_file(file_path):
                return f"Skipping file (unsupported): {file_path}"
            await self._print_message(f"[green]\nReading {file_path}[/green]")
            relative_path = os.path.relpath(file_path, root_folder) if root_folder else file_path
            if os.path.getsize(file_path) > self.max_file_size:
                content = await self._read_last_n_lines(file_path, self.max_lines)
            else:
                async with aiofiles.open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
                    content = await file.read()

            return f"--------- {relative_path} ---------\n" + content

        except Exception as e:
            return f"Error reading file {file_path}: {e}"

    def _is_safe_file(self, file_path):
        """Return True if the file has a whitelisted extension or is identified as text using python-magic."""
        if any(file_path.lower().endswith(ext) for ext in self.safe_extensions):
            return True  # Extension is whitelisted

        # For files not in the whitelist (or without an extension), use python-magic
        return self._is_text_file(file_path)

    def _is_text_file(self, file_path):
        """Determine if a file is text-based using python-magic."""
        try:
            mime = magic.Magic(mime=True)
            return mime.from_file(file_path).startswith("text")
        except Exception:
            return False


    async def _read_last_n_lines(self, file_path, num_lines):
            """Reads the last N lines of a file asynchronously using a buffered approach."""
            buffer = []
            loop = asyncio.get_running_loop()

            async with aiofiles.open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
                # Get file size using synchronous function in thread pool
                file_size = await loop.run_in_executor(None, lambda: self._get_file_size(file_path))
                pos = file_size
                data = ""

                while pos > 0 and len(buffer) < num_lines:
                    pos = max(0, pos - self.chunk_size)
                    await file.seek(pos)  # Seek is now async
                    chunk = await file.read(self.chunk_size)
                    data = chunk + data  # Prepend new chunk
                    lines = data.splitlines()

                    if len(lines) > num_lines:
                        buffer = lines[-num_lines:]
                        break
                    else:
                        buffer = lines

                return "\n".join(buffer) if buffer else "[File is empty]"

    def _get_file_size(self, file_path):
            """Gets file size synchronously for compatibility with aiofiles."""
            try:
                with open(file_path, "rb") as f:
                    f.seek(0, 2)  # Move to end of file
                    return f.tell()
            except Exception:
                return 0


    def generate_structure(self, folder_path, root_folder, prefix=""):
        """
        Generates a textual representation of the folder structure.
        All files within the folder are included, regardless of file type.
        """
        structure = f"{prefix}{os.path.basename(folder_path)}/\n"
        try:
            items = sorted(os.listdir(folder_path))
        except Exception as e:
            return f"Error reading folder {folder_path}: {e}"

        for item in items:
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path) and item not in self.ignore_folders and (self.scan_dot_folders or not item.startswith('.')):
                structure += self.generate_structure(item_path, root_folder, prefix + "--")
            elif os.path.isfile(item_path):
                relative_path = os.path.relpath(item_path, root_folder) if root_folder else item_path
                structure += f"{prefix}-- {relative_path}\n"

        return structure

    async def read_folder(self, folder_path, root_folder=None):
        """Recursively scans and reads all files in a folder.
           The folder structure is generated for all files; however, only files with safe extensions
           are attempted to be read (others are skipped).
        """
        await self._print_message(f"[green]\nOpening {folder_path}[/green]")
        if root_folder is None:
            root_folder = folder_path

        try:
            structure = self.generate_structure(folder_path, root_folder)
            file_contents = "\n### File Contents ###\n"
            
            for root, _, files in os.walk(folder_path):
                if any(ignored in root.split(os.sep) for ignored in self.ignore_folders):
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    if not any(file.lower().endswith(ext) for ext in self.safe_extensions):
                        file_contents += f"\nSkipping file (unsupported): {file_path}"
                    else:
                        content = await self.read_file(file_path, root_folder)
                        file_contents += f"\n{content.strip()}\n"
            return structure + file_contents

        except PermissionError:
            return f"Error: Permission denied to access '{folder_path}'."

    async def search_files(self, missing_path, search_dir=None, max_results=10):
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
                    if len(results) >= max_results:
                        return results
        return results

    async def prompt_search(self, missing_path):
        """
        Prompts the user to search for a file when the target is missing.
        Uses a popup with a radiolist if UI is available, falling back to terminal input.
        """
        while True:
            results = await self.search_files(missing_path)
            
            if not results:
                if hasattr(self, "ui") and self.ui is not None:
                    popup = RadiolistPopup(
                        title="No matches found",
                        text=f"No matches found for '{missing_path}'. Would you like to try again?",
                        options=[("yes", "Yes"), ("no", "No")]
                    )
                    self.ui.mount(popup)
                    retry = await popup.wait_for_choice()
                    popup.remove()
                else:
                    print(f"No matches found for '{missing_path}'.")
                    retry = input("Would you like to try again? (yes/no): ").strip().lower()
                if retry == "no":
                    return None
                if hasattr(self, "ui") and self.ui is not None:
                    missing_path = await self.ui.get_user_input("Modify search term:")
                else:
                    missing_path = input("Modify search term: ")
                continue

            options = [(res, res) for res in results] + [("cancel", "Cancel")]
            if hasattr(self, "ui") and self.ui is not None:
                popup = RadiolistPopup(
                    title="Select a file",
                    text=f"Multiple matches found for '{missing_path}'. Please choose one:",
                    options=options
                )
                self.ui.mount(popup)
                choice = await popup.wait_for_choice()
                popup.remove()
            else:
                print(f"Multiple matches found for '{missing_path}'. Please choose one:")
                for i, res in enumerate(results, start=1):
                    print(f"{i}. {res}")
                choice_str = input("Enter the number of your choice (or 'cancel'): ").strip()
                if choice_str.lower() == "cancel":
                    return None
                if choice_str.isdigit() and 1 <= int(choice_str) <= len(results):
                    choice = results[int(choice_str)-1]
                else:
                    print("Invalid input, try again.")
                    continue

            if choice == "cancel":
                return None
            return choice

    async def _print_message(self, message: str):
        """Print messages either through UI or terminal."""
        if self.ui:
            await self.ui.buffer.put(message)
        else:
            print(message)
