import sys
import asyncio
from utils.pipe_utils import PipeUtils
from utils.args_utils import parse_args
from chatbot.manager import ChatManager
from ollama_client.validator import validate_install
from utils.symlink_utils import create_symlink, remove_symlink


async def async_main():
    args = parse_args()

    if args.install:
        create_symlink()
        return
    if args.uninstall:
        remove_symlink()
        return
    if validate_install():
        chat_manager = ChatManager()
        pipe_utils = PipeUtils(chat_manager)

        user_input = args.prompt or args.string_input or ""
        file = args.file or ""
        pipe_content = ""
        stdin_piped = not sys.stdin.isatty()
        stdout_piped = not sys.stdout.isatty()

        if stdin_piped:
            pipe_content = await pipe_utils.read_pipe()
        if stdout_piped:
            chat_manager.ui = None
            response = await chat_manager.deploy_task(user_input, file, pipe_content)
            print(response)
        else:
            if chat_manager.ui:
                (
                    chat_manager.ui.user_input,
                    chat_manager.ui.file,
                    chat_manager.ui.file_content,
                ) = user_input, file, pipe_content

                await chat_manager.ui.run_async()


if __name__ == "__main__":
    asyncio.run(async_main())
