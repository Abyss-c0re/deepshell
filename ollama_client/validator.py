import os
import ast
import ollama
import subprocess
from utils.logger import Logger
from config.settings import DEFAULT_HOST


logger = Logger.get_logger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "settings.py")
REQUIRED_VERSION = "0.6.2"

def get_installed_version():
    """
    Check the installed Ollama version.
    """
    try:
        output = subprocess.check_output(["ollama", "--version"], text=True).strip()
        return output.split()[-1]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def ensure_ollama():
    """
    Check if Ollama is installed and up to date. If not, print instructions and return False.
    """
    installed_version = get_installed_version()

    if installed_version is None:
        print("Ollama is not installed. Run the following command to install it:")
        print("curl -fsSL https://ollama.com/install.sh | sh")
        return False

    if installed_version < REQUIRED_VERSION:
        print(f"Ollama version {installed_version} is outdated. Run the following command to update:")
        print("curl -fsSL https://ollama.com/install.sh | sh")
        return False

    logger.info(f"Ollama is installed and up to date (version {installed_version}).")
    return True

def extract_model_names(config_path=CONFIG_PATH):
    """
    Extract values from all *_MODEL variables in settings.py 
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file '{config_path}' does not exist.")

    with open(config_path, 'r', encoding='utf-8') as file:
        config_content = file.read()

    try:
        parsed_content = ast.parse(config_content)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in the configuration file: {e}")

    model_names = set()

    for node in ast.walk(parsed_content):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith('_MODEL'):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        model_names.add(node.value.value)  # Extract the assigned string value

    return model_names

def validate_install(config_path=CONFIG_PATH):
    """
    Ensure Ollama is installed and up to date before checking models.
    """

    if DEFAULT_HOST != "http://localhost:11434":
        return True

    if not ensure_ollama():
        return False

    model_names = extract_model_names(config_path)
    available_model_names = set()

    try:
        available_models = ollama.list()
        available_models = available_models.models
        for model in available_models:
            available_model_names.add(model.model)
    except Exception as e:
        raise RuntimeError(f"Failed to list models using Ollama: {e}")

    missing_models = model_names - available_model_names

    if not missing_models:
        logger.info("All required models are already available.")
        return True

    for model in missing_models:
        try:
            print(f"Pulling missing model: {model}")
            ollama.pull(model)
        except Exception as e:
            print(f"Failed to pull model '{model}': {e}")

    return True
