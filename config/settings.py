from enum import Enum, auto
from config.system_prompts import *

class Mode(Enum):
    DEFAULT = auto()
    ADVANCED = auto()
    CODE = auto()
    SHELL = auto()
    SYSTEM = auto()
    HELPER = auto ()
    VISION = auto()

# Ollama Settings
DEFAULT_HOST = "http://localhost:11434"

DEFAULT_MODEL = "phi4:14b"
CODE_MODEL = "deepseek-coder-v2:16b"
SHELL_MODEL = "granite3.2:8b"
SYSTEM_MODEL = "granite3.2:8b"
HELPER_MODEL = "deepseek-r1:1.5b"
VISION_MODEL = "minicpm-v:8b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Mapping Mode to Configuration
MODE_CONFIGS = {
    Mode.DEFAULT: {"model": DEFAULT_MODEL, "temp": 0.4, "prompt": "", "stream": True},
    Mode.CODE:    {"model": CODE_MODEL, "temp": 0.5, "prompt": CODE, "stream": True},
    Mode.SHELL:   {"model": SHELL_MODEL, "temp": 0.4, "prompt": SHELL, "stream": True},
    Mode.SYSTEM:  {"model": SYSTEM_MODEL, "temp": 0.5, "prompt": SYSTEM, "stream": True},
    Mode.HELPER:  {"model": HELPER_MODEL, "temp": 0.5, "prompt": "", "stream": False},
    Mode.VISION:  {"model": VISION_MODEL, "temp": 0.6, "prompt": "", "stream": False},
}

#Logging
LOG = True
LOG_LEVEL = "info" #Possible values: debug, info, warning, error, critical
LOG_TO_FILE = False
LOG_TO_UI = True

#Rendering
RENDER_DELAY = 0.0069 # Delay between rendering lines

#HistoryManager
MSG_THR = 0.5 # Simularity threshold for history
CONT_THR = 0.6 # Simularity threshold for content such as files and terminal output
NUM_MSG = 5 # Number of messages submitted to the chatbot from history
OFF_THR = 0.7 # Off-topic threshold
OFF_FREQ = 4 # Off-topic checking frequency (messages)
SLICE_SIZE = 4 # Last N messages to analyze for off-topic 

#ShellUtils Config
SHELL_TYPE = "/bin/bash"
MONITOR_INTERVAL = 60 # Timeout until when user will be prompted to abort command execution 
FINALIZE_OUTPUT = True # Output post-processing such as trimming
MAX_OUTPUT_LINES = 30000

#FileProcessing Config 
PROCESS_IMAGES = False # Turn this on if you want to get a description of the images
IMG_INPUT_RES = (512, 512)

IGNORE_DOT_FILES = False
MAX_FILE_SIZE = 6 * 1024 * 1024 #6MB
MAX_LINES = 30000

# General text formats
TEXT_FORMATS = [
    '.txt', '.md', '.json', '.csv', '.ini', '.cfg', '.xml', 
    '.yaml', '.yml', '.toml', '.log', '.sql', '.html', '.htm', 
    '.css', '.js', '.conf', '.properties', '.rst'
]

# Programming languages
PROGRAMMING_LANGUAGES = [
    '.py', '.c', '.cpp', '.h', '.hpp', '.java', '.cs', '.rs', '.go', 
    '.rb', '.php', '.sh', '.bat', '.pl', '.lua', '.swift', '.kt', '.m', 
    '.r', '.jl', '.dart', '.ts', '.v', '.scala', '.fs', '.asm', '.s', 
    '.vbs', '.ps1', '.clj', '.groovy', '.perl', '.f90', '.f95', '.ml'
]

# Image formats
IMAGE_FORMATS = [
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'
]

SUPPORTED_EXTENSIONS = TEXT_FORMATS + PROGRAMMING_LANGUAGES + IMAGE_FORMATS


IGNORED_FOLDERS =  ['__pycache__', '.git', '.github', '.svn', '.hg', 'Android', 'android-studio', 'miniconda3']
