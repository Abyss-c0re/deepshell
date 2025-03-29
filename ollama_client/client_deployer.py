import sys
from config.settings import *
from utils.args_utils import parse_args 
from ollama_client.api_client import OllamaClient

class ClientDeployer:

    def __init__(
            self,
            mode: Mode | None = None
    ):
        self.args = parse_args()
        self.user_input = self.args.prompt or self.args.string_input or None
        self.file = self.args.file

        if mode:
            self.mode = mode
        else:
            self.mode = (
                Mode.SHELL if self.args.shell else
                Mode.CODE if self.args.code else
                Mode.SYSTEM if self.args.system else
                Mode.DEFAULT
            )
        
        config = MODE_CONFIGS[self.mode]
        self.host = DEFAULT_HOST
        self.model = config["model"]
        self.config = self.generate_config(temp=config["temp"], prompt=config["prompt"])
        self.stream = config["stream"]

    def deploy(self)-> OllamaClient:
        """
        Deploys an isntance of Ollama API Client
        """

        if self.args.host:
            self.host = self.args.host
        if self.args.model:
            self.model = self.args.model

        return OllamaClient(
            host=self.host,
            model=self.model,
            config=self.config,
            mode=self.mode,  
            stream=self.stream,
            render_output=sys.stdout.isatty(),
            show_thinking=self.args.thinking
        )

    def generate_config(
            self, 
            temp=0.7, 
            prompt=""
    )-> dict:
        return {"temperature": temp, "system": prompt}

