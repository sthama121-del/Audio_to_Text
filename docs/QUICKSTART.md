# 🚀 Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install Dependencies (One-Time Setup)

Open your terminal and run:

```bash
pip install streamlit openai-whisper torch torchvision torchaudio
```

**Or use the requirements file:**
```bash
pip install -r requirements.txt
```

## Step 2: Install ffmpeg

### Windows (using Chocolatey):
```bash
choco install ffmpeg
```

### macOS (using Homebrew):
```bash
brew install ffmpeg
```

### Linux (Ubuntu/Debian):
```bash
sudo apt update && sudo apt install ffmpeg
```

## Step 3: Run the App

```bash
streamlit run audio_transcriber_app.py
```

That's it! The app will open in your browser automatically.

## Step 4: Transcribe Your Audio

1. Click "Browse files" to upload your audio file (MP3, WAV, M4A, etc.)
2. Choose your preferred model size in the sidebar (use "base" for best balance)
3. Select your output format
4. Click "Transcribe Audio"
5. Wait for the transcription to complete
6. Download your transcript as a .txt file!

## ⚡ Pro Tips

- **First time?** The app will download the model on first run (takes a few minutes)
- **Large files?** Use the "base" or "small" model for faster processing
- **Speaker detection?** Enable "Simple Speaker Detection" for interviews/podcasts
- **Best quality?** Use the "small" or "medium" model for higher accuracy

## 🆘 Getting Help

If the app doesn't start:
1. Make sure all dependencies are installed
2. Verify ffmpeg is installed: run `ffmpeg -version`
3. Try restarting your terminal
4. Check the full README.md for detailed troubleshooting

---

**Need more help?** Check the full README.md file for detailed instructions and troubleshooting.
