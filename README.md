# Audio to Text Transcription Engine

> Multilingual audio transcription with AI-powered formatting — built for Telugu devotional content, designed for any language.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Problem Statement

Devotional and religious audio content — bhajans, discourses, kirtans — is widely consumed but rarely available in readable, searchable text format. Existing cloud transcription APIs charge per-minute fees that become prohibitive at scale, and lack fidelity for South Indian languages like Telugu, Tamil, and Kannada.

This project delivers a fully local, cost-free transcription solution that:
- Handles Telugu, Hindi, Tamil, and other South Asian languages
- Accepts local file uploads (MP3, WAV, M4A, OGG, FLAC)
- Runs entirely on-premise — no API keys, no cloud costs, unlimited usage

---

## Architecture

Input Layer
  Local file upload (MP3, WAV, M4A, OGG, FLAC)
          |
          v
Audio Processing
  librosa + soundfile — normalization, silence trimming, 16kHz conversion
          |
          v
Transcription Engine
  OpenAI Whisper (local model)
  - Auto language detection
  - Supports Telugu, Hindi, Tamil, English and 90 more languages
          |
          v
Formatting Layer
  - Plain Text with bullets
  - Timestamped transcript
  - Simple Speaker Detection (Q&A format)
          |
          v
Output
  Downloadable .txt transcript

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI Framework | Streamlit | Web interface for upload and configuration |
| Transcription | OpenAI Whisper (local) | Speech-to-text, 90+ languages |
| Audio Processing | librosa + soundfile | Normalization, silence trimming |
| Language | Python 3.9+ | Core runtime |

---

## Getting Started

### Prerequisites

Ensure ffmpeg is installed on your system.

macOS:
  brew install ffmpeg

Ubuntu/Debian:
  sudo apt install ffmpeg

Windows:
  Download from https://ffmpeg.org/download.html

### Installation

Clone the repository:
  git clone https://github.com/YOUR-USERNAME/audio-to-text.git
  cd audio-to-text

Create virtual environment:
  python -m venv .venv

Activate on macOS/Linux:
  source .venv/bin/activate

Activate on Windows:
  .venv\Scripts\Activate.ps1

Install PyTorch CPU first:
  pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu

Install remaining dependencies:
  pip install -r requirements.txt

### Running the Application

  streamlit run app.py

Navigate to http://localhost:8501 in your browser.

---

## Usage

1. Upload — Select an audio file (MP3, WAV, M4A, OGG, FLAC)
2. Configure — Choose model size and output format in the sidebar
3. Transcribe — Click Transcribe Audio and wait for processing
4. Download — Save the transcript as a .txt file

---

## Output Formats

| Format | Best For |
|--------|----------|
| Plain Text with Bullets | Reading, sharing, printing |
| With Timestamps | Reference, editing, subtitles |
| Simple Speaker Detection | Q&A sessions, interviews, discourse |

---

## Model Size Guide

| Model | Speed | Accuracy | RAM Required |
|-------|-------|----------|-------------|
| tiny | Fastest | Basic | ~1 GB |
| base | Fast | Good | ~1 GB |
| small | Moderate | Better | ~2 GB |
| medium | Slow | High | ~5 GB |
| large | Slowest | Best | ~10 GB |

---

## Business Value

| Dimension | Impact |
|-----------|--------|
| Cost | $0 per transcription vs $0.006-$0.024/min on cloud APIs |
| Privacy | Audio never leaves local machine |
| Scale | Unlimited transcriptions, no rate limits |
| Language support | 90+ languages including Telugu, Hindi, Tamil |

---

## Future Enhancements

- YouTube ingestion — download and transcribe directly from YouTube URLs via yt-dlp
- Batch processing — transcribe entire folders of audio files
- Speaker diarization — accurate speaker identification using pyannote.audio
- Translation layer — auto-translate transcripts to English
- PDF export — formatted output with Unicode font support
- REST API — expose transcription as a microservice endpoint

---

## Known Limitations

- Transcription quality depends on audio clarity and background noise
- Whisper large model requires ~10GB RAM
- Speaker detection uses pause heuristics — not true diarization

---

## Project Structure

audio-to-text/
  app.py                    Main Streamlit application
  requirements.txt          Python dependencies
  requirements.cpu.txt      PyTorch CPU install instructions
  Dockerfile                Container definition
  run.sh                    Shell launch script
  README.md                 This file
  .gitignore                Git exclusions
  docs/
    screenshots/            UI screenshots
    sample_output/          Sample transcription outputs
  archive/                  Backup and older versions

---

## Author

Srikanth — Senior Data Engineer / AI Engineer
Specializing in lakehouse architecture, GenAI pipelines, and multilingual NLP systems.

---

## License

MIT License — free to use, modify, and distribute.
