#!/bin/bash
# =============================================================
# setup.sh — One-shot setup for RAG_Eval_Training
# Run this FROM your project folder:
#   cd /Users/Sri/Documents/Audio_to_Text/RAG_Eval_Training
#   bash setup.sh
# =============================================================

echo "============================================="
echo " RAG_Eval_Training — Project Setup"
echo "============================================="
echo ""

# --- Step 1: Unzip files.zip into current directory ---
if [ -f "files.zip" ]; then
    echo "[1/5] Unzipping files.zip..."
    unzip -o files.zip
    echo "      ✓ Done"
else
    echo "[1/5] files.zip not found — skipping unzip."
    echo "      (Files may already be extracted.)"
fi
echo ""

# --- Step 2: Verify all required files exist ---
echo "[2/5] Verifying project files..."
REQUIRED_FILES=(
    "main.py"
    "config.py"
    "ingestion.py"
    "retrieval.py"
    "generation.py"
    "evaluation.py"
    "report.py"
    "hr_policy.txt"
    "requirements.txt"
    ".env.example"
)

ALL_GOOD=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "      ✓ $f"
    else
        echo "      ✗ MISSING: $f"
        ALL_GOOD=false
    fi
done

if [ "$ALL_GOOD" = false ]; then
    echo ""
    echo "  !! Some files are missing. Check the zip contents."
    exit 1
fi
echo ""

# --- Step 3: Copy .env.example to .env (if .env doesn't already exist) ---
echo "[3/5] Setting up .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "      ✓ Created .env from .env.example"
    echo "      ⚡ YOU MUST EDIT .env and add your API keys (see Step 4 below)"
else
    echo "      ✓ .env already exists — skipping."
fi
echo ""

# --- Step 4: Install Python dependencies into the active venv ---
echo "[4/5] Installing Python dependencies..."
pip install -r requirements.txt
echo "      ✓ Dependencies installed"
echo ""

# --- Step 5: Final status ---
echo "[5/5] Setup complete! Here is your project:"
echo ""
ls -la *.py *.txt *.example .env 2>/dev/null
echo ""
echo "============================================="
echo " NEXT STEPS:"
echo "============================================="
echo ""
echo "  1. Open .env in your editor and add your keys:"
echo "       OPENAI_API_KEY=sk-your-openai-key-here"
echo "       ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here"
echo ""
echo "  2. Run the pipeline:"
echo "       python main.py"
echo "============================================="
