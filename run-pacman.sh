#!/usr/bin/env bash
# Launch OBSERVE: Agentic Ghosts Intelligence (Pacman ML)
# Usage: ./run-pacman.sh [optional: --ml --model models/ghost_agent/best_model.zip]

set -e
cd "$(dirname "$0")"

# Use venv if present
if [ -d "venv" ]; then
  source venv/bin/activate
fi

exec python3 main.py "$@"
