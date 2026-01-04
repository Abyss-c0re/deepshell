import platform
import re
from datetime import datetime
from utils.logger import Logger

logger = Logger.get_logger()


def _get_distro_info() -> str:
    """
    Detects the actual Linux distribution and returns a user-friendly string.
    Falls back to platform.uname() if detection fails.
    """
    try:
        with open("/etc/os-release", "r") as f:
            content = f.read()

        # Extract PRETTY_NAME for nice display, fallback to ID + VERSION_ID
        pretty_name = re.search(r'^PRETTY_NAME="?(.+?)"?$', content, re.MULTILINE)
        if pretty_name:
            return pretty_name.group(1)

        id_match = re.search(r'^ID="?(.+?)"?$', content, re.MULTILINE)
        version_match = re.search(r'^VERSION_ID="?(.+?)"?$', content, re.MULTILINE)
        distro_id = id_match.group(1) if id_match else "unknown"
        version = f" {version_match.group(1)}" if version_match else ""
        return f"{distro_id.capitalize()}{version} Linux"

    except Exception as e:
        logger.debug(f"Distro detection failed: {e}, falling back to uname")
        return f"{platform.system()} {platform.release()} ({platform.machine()})"


class PromptHelper:
    """
    A utility class for generating shell command prompts and analyzing command output.
    Now with proper distro detection!
    """

    user_system = _get_distro_info()
    current_time = datetime.now().isoformat()

    @staticmethod
    def shell_helper(user_input: str) -> str:
        """
        Generates a shell command prompt tailored to the actual distro (Arch, Debian, etc.).
        """
        return f"""
Generate a SINGLE, non-interactive shell command for this system: **{PromptHelper.user_system}**

Rules:
- Use the correct package manager (pacman for Arch, apt for Debian/Ubuntu, dnf for Fedora, zypper for openSUSE, etc.).
- Never explain, never wrap in code blocks, never add extra text.
- If the command needs sudo, include it.
- Always use flags to make it non-interactive (-y, --noconfirm, --assume-yes, etc.).
- Do NOT ask for confirmation.

User wants to: {user_input}
""".strip()

    @staticmethod
    def analyzer_helper(command: str, output: str) -> str:
        return f"""
Analyze the output of the following command: {command}

Output:
{output}

Summarize errors, warnings, success status, and key information.
Only mention system details if relevant to the issue.

SYSTEM INFO (use only if needed):
Time: {PromptHelper.current_time}
OS: {PromptHelper.user_system}
""".strip()

    @staticmethod
    def topics_helper(history: list) -> str:
        history_text = str(history)
        logger.debug(f"Topics helper: injected history: {history_text}")
        return f"""
Based on the following conversation history, please name a topic and provide a description in JSON format:

{history_text}

Response must be valid JSON with keys:
- "topic_name": string
- "topic_description": string
""".strip()

    @staticmethod
    def analyze_code(content: str) -> str:
        return f"""
Analyze the following code and return structured metadata in JSON format with these keys:
- "functions": list of {{ "name": str, "description": str }}
- "classes": list of {{ "name": str, "purpose": str }}
- "purpose": brief overall summary

Code:
{content}
""".strip()