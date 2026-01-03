# Default to CPU/NVIDIA variant (change with --build-arg OLLAMA_VARIANT=rocm for AMD)
ARG OLLAMA_VARIANT=latest
FROM ollama/ollama:${OLLAMA_VARIANT}

# Install required packages (Python, pip, curl, libmagic1 for python-magic)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Link python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Working directory
WORKDIR /app

# Copy DeepShell files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Ollama port
EXPOSE 11434

# Custom entrypoint (starts ollama serve in background + DeepShell)
ENTRYPOINT ["/entrypoint.sh"]
