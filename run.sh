#!/usr/bin/env bash
set -e

echo "Starting Streamlit Audio Transcriber..."

streamlit run audio_transcriber_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true
