#!/usr/bin/env bash
# Pull the local models. Run once after installing Ollama (https://ollama.com).
set -e

# Primary extraction model. ~9 GB at Q4 — fits a 16 GB GPU with room for context.
ollama pull qwen2.5:14b-instruct

# Lighter fallback for 8 GB cards or faster iteration (~4.7 GB). Uncomment to use,
# and set GAP_LLM_MODEL=qwen2.5:7b-instruct in your .env.
# ollama pull qwen2.5:7b-instruct

echo "Done. Verify with:  python main.py --check"
