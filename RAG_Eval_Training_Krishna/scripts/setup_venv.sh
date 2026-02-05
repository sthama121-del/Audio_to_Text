#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp -n .env.example .env || true

echo ""
echo "✅ Setup complete."
echo "Next:"
echo "  1) edit .env with your keys"
echo "  2) python ingest.py"
echo "  3) python main.py"
