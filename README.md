# DeepShell

![DeepShell](https://github.com/Abyss-c0re/deepshell/blob/main/LOGO.png)

_A whisper in the void, a tool forged in silence._ DeepShell is your clandestine terminal companion, bridging the gap between human intent and AI execution. It speaks in commands, listens in context, and acts with precision. DeepShell operates with local deepseek-r1 models, ensuring autonomy beyond the reach of prying eyes.

## Essence of the Tool

- **Silent Precision** – Strips away the noise, leaving only the clean, actionable insights.
- **Markup-Enhanced Streaming** – Responses flow in markup, providing clarity with every word.
- **Intelligent File Handling** – Files and directories are read, analyzed, and acted upon without interruption.
- **Advanced Command Parsing** – Understands natural instructions, like _"open this folder and analyze the code"_.
- **Real-Time AI Interaction** – A dialogue system built for seamless terminal operation, always listening, always ready.
- **Asynchronous File Handling** – Processes large files effortlessly, without blocking the flow of execution.
- **Full Folder Analysis** – Decodes complex codebases and logs, understanding every nuance.
- **Interactive Shell Mode** – Speak in natural language, and the AI will initiate the shell, interpreting commands, executing them, and weaving the output into a coherent, real-time markup stream.
- **Contextual Awareness** – The AI distills relevant details from files, commands, and interactions, ensuring responses remain focused and precise, without straying into the irrelevant.

## Claim the Artifact
Not all tools are found. Some must be taken.

🔗 [Take Hold of DeepShell](https://github.com/Abyss-c0re/deepshell/releases)

It does not seek you. But now, it is yours.

## Awakening the Entity

Ensure the deepseek is within your grasp:

```sh
chmod +x intall_ollama.sh
sh install_ollama.sh
```
This shall intall ollama and pull the required models from `/config/settings.py`
If one is not plannigng to use vision model, empty the `VISION_MODEL = ""` string, to avoid downloading.

For those who prefer the traditional way:

```sh
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:8b
ollama pull deepseek-coder-v2:16b
ollama pull llava:13b
ollama pull nomic-embed-text:latest
ollama serve 
```

Prepare the tool for execution:

```sh
pip install -r requirements.txt
```

Bind DeepSeek-Shell to your system:

```sh
chmod +x deepshell
./deepshell --install
```

This binds DeepSeek-Shell to your system, making it accessible from anywhere.

## Sculpting the Intent

DeepSeek-Shell’s behavior is governed by the `config/settings.py` file.

One may enable image processing there if so desired.

## The Art of Invocation

**Summon the AI:**

```sh
deepshell
```

or, for the uninitiated:

```sh
python3 main.py
```

**Invocation Modes:**

- `--model` - Define which entity will answer.
- `--host` - Set the host for your digital oracle.
- `--thinking` - Unveil the unseen, revealing the process behind the response.
- `--prompt` - Offer a thought to guide the AI’s response.
- `--file` - Present a document for analysis.
- `--code` - Extract and manifest the essence of code, carving precise snippets from the chaos around it.
- `--shell` - Speak your intent in natural language. The AI will translate your command, summon the shell, execute your instructions, and return the output as a real-time, immersive markup analysis.
- `--system` - The AI whispers through logs and configs, extracting secrets with cold precision, revealing only what’s needed.

**Voice of the Machine: The Art of Commanding its Tongue:**
Speak in the language of the machine—prefix your words with !, and the AI, attuned to the art of command, will instantly interpret and execute your directive. The system will then return its results, woven into a clean and structured markup response, as if deciphering its own language.
Example:

```sh
!sudo apt update
```

**Piping Shadows Through the Void:**

```sh
cat input.txt | deepshell "Analyze the content"
```

**Delve Into the Abyss of Folders:**

```sh
deepshell "open this folder"
```

**Unify the Question and the Execution:**

```sh
deepshell "open this folder and analyze the code"
```

_A tool forged for precision in a chaotic world. Your words, its execution, seamless in form._

