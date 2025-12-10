# DeepShell

![DeepShell](https://github.com/Abyss-c0re/deepshell/blob/main/LOGO.png)

_A whisper in the void, a tool forged in silence._ DeepShell is your clandestine terminal companion, bridging the gap between human intent and AI execution. It speaks in commands, listens in context, and acts with precision.

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

[**Take Hold of DeepShell**](https://github.com/Abyss-c0re/deepshell/releases)

_It does not seek you. But now, it is yours._

## Awakening the Entity

Prepare the tool for execution:

```sh
pip install -r requirements.txt
```
Bind DeepShell to your system:

```sh
chmod +x deepshell
./deepshell --install
```
This binds DeepShell to your system, making it accessible from anywhere.

## Sculpting the Intent

[**Requires configuration**](https://github.com/Abyss-c0re/deepshell/wiki#configuration)

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

** The Vessel of Containment: Docker **

DeepShell may be bound within a Docker vessel, self-contained and portable, with Ollama as its inner oracle.

### Forging the Vessel

Build the image, choosing your hardware's essence:

```sh
# CPU only (or NVIDIA fallback)
docker build -t deepshell .

# AMD GPU (ROCm acceleration)
docker build --build-arg OLLAMA_VARIANT=rocm -t deepshell .
```

**Summoning the Vessel**

CPU

```sh
docker run -it --rm \
  -v /:/host:ro \
  -p 11434:11434 \
  deepshell
```
NVIDIA GPU (requires NVIDIA Container Toolkit installed and Docker restarted)

```sh
docker run -it --rm \
  -v /:/host:ro \
  -p 11434:11434 \
  --gpus all \
  deepshell
```
AMD GPU (ROCm)

```sh
docker run -it --rm \
  -v /:/host:ro \
  -p 11434:11434 \
  --device /dev/kfd \
  --device /dev/dri \
  deepshell
```

**Echoes in the Abyss: The Neovim Conduit**

The [DeepShell.nvim](https://github.com/Abyss-c0re/deepshell-nvim) plugin serves as the unseen bridge between the arcane depths of Neovim and the profound intelligence of DeepShell.



_A tool forged for precision in a chaotic world. Your words, its execution, seamless in form._



