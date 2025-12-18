from typing import Tuple
from utils.logger import Logger
from config.settings import Mode
from pipeline.pipe_filter import PipeFilter
from ollama_client.api_client import OllamaClient
from ollama_client.client_deployer import ClientDeployer

logger = Logger.get_logger()


def deploy_chatbot(mode: Mode | None = None) -> Tuple[OllamaClient, PipeFilter]:
    """
    Deploys a chatbot with an optional mode.
    """
    client_deployer = ClientDeployer(mode)
    chatbot = client_deployer.deploy()
    filter = PipeFilter(chatbot)

    return chatbot, filter
