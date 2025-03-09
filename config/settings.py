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

DEFAULT_MODEL = "deepseek-r1:8b"
CODE_MODEL = "deepseek-coder-v2:16b"
SHELL_MODEL = "deepseek-coder-v2:16b"
SYSTEM_MODEL = "deepseek-r1:8b"
HELPER_MODEL = "deepseek-r1:1.5b"
VISION_MODEL = "llava:7b"
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
LOGGING = True
LOGGING_LEVEL = "info"

#Rendering
RENDER_DELAY = 0.01 # Delay between rendering lines

#HistoryManager
MSG_THR = 0.5 # Simularity threshold for history
CONT_THR = 0.6 # Simularity threshold for content such as files and terminal output
NUM_MSG = 5 # Number of messages submitted to the chatbot from history
OFF_THR = 0.7 # Off-topic threshold
OFF_FREQ = 4 # Off-topic checking frequency  (messages)
SLICE_SIZE = 4 # Last N messages to analyze for off-topic 

#ShellUtils Config
SHELL_TYPE = "/bin/bash"
MONITOR_INTERVAL = 60 # Timeout until when user will be prompted to abort command execution 
FINALIZE_OUTPUT = True # Output post-processing such as trimming
MAX_OUTPUT_LINES = 1000


#FileProcessing Config 
PROCESS_IMAGES = False # Turn this on if you want to get a description of the images
IMG_INPUT_RES = (672, 672)

MAX_FILE_SIZE = 6 * 1024 * 1024 #6MB
MAX_LINES = 1000
CHUNK_SIZE = 4000

SUPPORTED_EXTENSIONS = [
    # General text formats
    '.txt', '.md', '.json', '.csv', '.ini', '.cfg', '.xml',  
    '.yaml', '.yml', '.toml', '.log', '.sql', '.html', '.htm',  
    '.css', '.js', '.conf', '.properties', '.rst',

    # Programming languages
    '.py', '.c', '.cpp', '.h', '.hpp', '.java', '.cs', '.rs', '.go',  
    '.rb', '.php', '.sh', '.bat', '.pl', '.lua', '.swift', '.kt', '.m',  
    '.r', '.jl', '.dart', '.ts', '.v', '.scala', '.fs', '.asm', '.s',  
    '.vbs', '.ps1', '.clj', '.groovy', '.perl', '.f90', '.f95', '.ml',
    
    # Image formats
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'
]

IGNORED_FOLDERS =  ['__pycache__', '.git', '.svn', '.hg', 'Android', 'android-studio', 'miniconda3']
