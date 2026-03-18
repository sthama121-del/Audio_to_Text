# Audio to Text Transcription Engine

> Multilingual audio transcription with AI-powered formatting — built for Telugu devotional content, designed for any language.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)](https://github.com/openai/whisper)
[![yt-dlp](https://img.shields.io/badge/yt--dlp-YouTube-red)](https://github.com/yt-dlp/yt-dlp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Problem Statement

Devotional and religious audio content — bhajans, discourses, kirtans — is widely consumed but rarely available in readable, searchable text format. Existing cloud transcription APIs charge per-minute fees that become prohibitive at scale, and lack fidelity for South Indian languages like Telugu, Tamil, and Kannada.

This project delivers a fully local, cost-free transcription solution that:
- Handles Telugu, Hindi, Tamil, Kannada, and other South Asian languages
- Accepts local file uploads AND transcribes directly from YouTube URLs
- Supports forced language selection to eliminate hallucination on South Asian content
- Runs entirely on-premise — no API keys, no cloud costs, unlimited usage

---

## Architecture

Input Layer
  Option A: Local file upload (MP3, WAV, M4A, OGG, FLAC)
  Option B: YouTube URL  ->  yt-dlp downloads audio locally  ->  temp MP3
          |
          v
Audio Processing
  librosa + soundfile — normalization, silence trimming, 16kHz conversion
          |
          v
Transcription Engine
  OpenAI Whisper (local model)
  - Forced language selection (Telugu, Hindi, Tamil, Kannada, English)
  - Auto language detection fallback
  - Optimized settings for South Asian speech and music
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
  Audio never leaves your machine — zero cloud dependency

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI Framework | Streamlit | Web interface with tabbed layout |
| Transcription | OpenAI Whisper (local) | Speech-to-text, 90+ languages |
| YouTube Download | yt-dlp | Local audio extraction from YouTube |
| Audio Processing | librosa + soundfile | Normalization, silence trimming |
| Language Selection | Telugu, Hindi, Tamil, Kannada, English | Force language for better accuracy |
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
  git clone https://github.com/sthama121-del/Audio_to_Text.git
  cd Audio_to_Text

Create virtual environment:
  python -m venv .venv

Activate on Windows:
  .venv\Scripts\Activate.ps1

Activate on macOS/Linux:
  source .venv/bin/activate

Install PyTorch CPU first:
  pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu

Install remaining dependencies:
  pip install -r requirements.txt

### Running the Application

  streamlit run app.py

Navigate to http://localhost:8501 in your browser.

---

## Usage

### Tab 1 — Upload Audio File
1. Select an audio file (MP3, WAV, M4A, OGG, FLAC)
2. Choose model size and language in the sidebar
3. Choose output format in the sidebar
4. Click Transcribe Audio
5. Download the transcript as .txt

### Tab 2 — YouTube URL
1. Paste any public YouTube URL
2. Choose model size and language in the sidebar
3. Click Download and Transcribe
4. Audio is downloaded locally, transcribed, then temp file is deleted
5. Download the transcript as .txt

### Language Selection (Sidebar)
- Select Telugu, Hindi, Tamil, Kannada, or English for best accuracy
- Use Auto-detect only if the language is unknown
- Forcing the language eliminates hallucination loops on South Asian content

---

## Output Formats

| Format | Best For |
|--------|----------|
| Plain Text with Bullets | Reading, sharing, printing |
| With Timestamps | Reference, editing, subtitles |
| Simple Speaker Detection | Q&A sessions, interviews, discourse |

---

## Model Size Guide

| Model | Speed | Accuracy | RAM Required | Recommended For |
|-------|-------|----------|-------------|-----------------|
| tiny | Fastest | Basic | ~1 GB | Quick tests |
| base | Fast | Good | ~1 GB | English content |
| small | Moderate | Better | ~2 GB | Telugu/Hindi speech |
| medium | Slow | High | ~5 GB | Telugu/Hindi songs |
| large | Slowest | Best | ~10 GB | Maximum accuracy |

---

## Business Value

| Dimension | Impact |
|-----------|--------|
| Cost | $0 per transcription vs $0.006-$0.024/min on cloud APIs |
| Privacy | Audio never leaves local machine |
| Scale | Unlimited transcriptions, no rate limits |
| Language accuracy | Forced language selection eliminates South Asian hallucinations |
| YouTube support | Transcribe any public video without downloading manually |

---

## Future Enhancements

- Batch processing — transcribe entire YouTube playlists or folders
- Speaker diarization — accurate speaker identification using pyannote.audio
- Translation layer — auto-translate Telugu/Hindi transcripts to English
- PDF export — formatted output with Unicode font support
- REST API — expose transcription as a microservice endpoint
- Audio type toggle — separate optimized settings for songs vs speech

---

## Known Limitations

- Transcription quality depends on audio clarity and background noise
- Heavy background music reduces accuracy even with forced language
- Whisper large model requires ~10GB RAM
- Speaker detection uses pause heuristics — not true diarization
- Only works with public YouTube videos

---

## Project Structure

Audio_to_Text/
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
