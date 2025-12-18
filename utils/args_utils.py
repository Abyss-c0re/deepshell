import argparse


def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Ollama Chat Mode")
    parser.add_argument("--model", type=str, default="", help="Specify the AI model")
    parser.add_argument(
        "--host", type=str, default="", help="Specify the Ollama API host"
    )
    parser.add_argument("--prompt", type=str, default="", help="Chat message")
    parser.add_argument("--file", type=str, help="File to include in chat")
    parser.add_argument(
        "string_input", nargs="?", type=str, help="Optional string input"
    )

    symlink_group = parser.add_mutually_exclusive_group()
    symlink_group.add_argument(
        "--install", action="store_true", help="Install symlink for deepshell"
    )
    symlink_group.add_argument(
        "--uninstall", action="store_true", help="Uninstall symlink for deepshell"
    )

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--code", action="store_true", help="Generate code only")
    output_group.add_argument(
        "--shell", action="store_true", help="Generate shell command"
    )
    output_group.add_argument(
        "--system", action="store_true", help="System Administration"
    )
    output_group.add_argument(
        "--thinking", action="store_true", help="Show thinking sections"
    )

    return parser.parse_args()
