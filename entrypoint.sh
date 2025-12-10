#!/bin/bash

# Start Ollama completely silently — no logs at all
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

# Wait quietly for startup
sleep 15

# Only DeepShell output will appear
exec python main.py "$@"
