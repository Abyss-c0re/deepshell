#!/bin/bash

echo "Starting Ollama server in the background..."
ollama serve &
OLLAMA_PID=$!

echo "Waiting 15 seconds for Ollama to fully initialize..."
sleep 15

echo "Ollama is ready! Starting DeepShell..."

# Run DeepShell, passing any command-line arguments (e.g., --shell)
exec python main.py "$@"

# Keep the container running even if DeepShell exits (optional, but safe)
wait $OLLAMA_PID
