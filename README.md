# 🎙️ Audio Transcription & Speaker Diarization App

A powerful Streamlit web application that transcribes audio files (podcasts, interviews, meetings) using OpenAI's Whisper model with optional speaker detection.

## ✨ Features

- **Multi-format Support**: Upload MP3, WAV, M4A, OGG, and FLAC files
- **Automatic Transcription**: Uses OpenAI Whisper for accurate speech-to-text
- **Multiple Output Formats**:
  - Plain text transcription
  - Timestamped transcription
  - Simple speaker detection (Speaker A, Speaker B, etc.)
- **Downloadable Results**: Export transcripts as .txt files
- **User-Friendly Interface**: Clean, intuitive Streamlit UI
- **Model Selection**: Choose from tiny to large models based on your needs

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM (recommended for base/small models)
- ffmpeg (for audio processing)

## 🚀 Installation Instructions

### Step 1: Install Python Dependencies

Open your terminal/command prompt and run:

```bash
# Install required packages
pip install streamlit openai-whisper torch torchvision torchaudio

# For additional audio format support
pip install ffmpeg-python
```

**Note for different operating systems:**

#### Windows:
```bash
pip install streamlit openai-whisper torch torchvision torchaudio
```

#### macOS:
```bash
pip install streamlit openai-whisper torch torchvision torchaudio
```

#### Linux:
```bash
pip install streamlit openai-whisper torch torchvision torchaudio
```

### Step 2: Install ffmpeg

**Windows:**
1. Download ffmpeg from https://ffmpeg.org/download.html
2. Extract and add to your system PATH

Or use chocolatey:
```bash
choco install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

### Step 3: Verify Installation

```bash
python -c "import whisper; print('Whisper installed successfully!')"
python -c "import streamlit; print('Streamlit installed successfully!')"
ffmpeg -version
```

## 🎯 How to Run the App

### Method 1: Direct Run

1. Save the `audio_transcriber_app.py` file to your computer
2. Open terminal/command prompt
3. Navigate to the directory containing the file:
   ```bash
   cd path/to/your/directory
   ```
4. Run the app:
   ```bash
   streamlit run audio_transcriber_app.py
   ```
5. Your default browser will open automatically with the app!

### Method 2: Using Python

```bash
python -m streamlit run audio_transcriber_app.py
```

The app will open in your browser at: `http://localhost:8501`

## 📖 How to Use

1. **Upload Audio File**: Click "Browse files" and select your audio file
2. **Choose Settings** (in sidebar):
   - Select model size (base recommended for most users)
   - Choose output format
   - Set number of speakers (if using speaker detection)
3. **Transcribe**: Click the "Transcribe Audio" button
4. **Wait**: Processing may take a few minutes depending on file size and model
5. **Download**: Click "Download Transcript" to save your transcription

## 🎨 Model Size Guide

| Model  | Size  | Speed    | Accuracy | Use Case |
|--------|-------|----------|----------|----------|
| tiny   | 39M   | Fastest  | Good     | Quick drafts |
| base   | 74M   | Fast     | Better   | **Recommended** |
| small  | 244M  | Medium   | Very Good| High quality |
| medium | 769M  | Slow     | Excellent| Professional |
| large  | 1550M | Slowest  | Best     | Maximum accuracy |

## ⚙️ Advanced Features

### True Speaker Diarization

For more accurate speaker identification, install pyannote.audio:

```bash
pip install pyannote.audio
```

You'll also need:
1. A Hugging Face account (free): https://huggingface.co/join
2. Accept the model terms: https://huggingface.co/pyannote/speaker-diarization
3. Create an access token: https://huggingface.co/settings/tokens

## 🔧 Troubleshooting

### Common Issues:

**Issue**: "No module named 'whisper'"
```bash
pip install --upgrade openai-whisper
```

**Issue**: "ffmpeg not found"
- Ensure ffmpeg is installed and added to your system PATH
- Restart your terminal after installation

**Issue**: CUDA/GPU errors
```bash
# Install CPU-only version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Issue**: Model download is slow
- The first run downloads the model (can take time)
- Models are cached for future use
- Check your internet connection

**Issue**: Out of memory
- Use a smaller model (tiny or base)
- Process shorter audio files
- Close other applications

## 📊 Performance Tips

- **For fast results**: Use 'tiny' or 'base' model
- **For accuracy**: Use 'small' or 'medium' model
- **For long files**: Consider splitting into chunks
- **GPU support**: Install CUDA-enabled PyTorch for faster processing

## 🤝 Support & Contribution

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure your audio file is in a supported format

## 📝 Example Workflow

```bash
# 1. Install dependencies
pip install streamlit openai-whisper torch

# 2. Navigate to app directory
cd /path/to/app

# 3. Run the app
streamlit run audio_transcriber_app.py

# 4. Open browser to localhost:8501

# 5. Upload your audio file

# 6. Click "Transcribe Audio"

# 7. Download your transcript!
```

## 🎓 Technical Details

- **Whisper Models**: Powered by OpenAI's Whisper ASR system
- **Framework**: Built with Streamlit for rapid prototyping
- **Audio Processing**: Uses ffmpeg for format conversion
- **Speaker Detection**: Simple pause-based heuristic (optional pyannote.audio for advanced)

## 📄 License

This application uses:
- OpenAI Whisper (MIT License)
- Streamlit (Apache 2.0 License)

## 🚨 Important Notes

- First run will download the selected Whisper model (can take several minutes)
- Larger audio files require more processing time
- The simple speaker detection is heuristic-based and may not be 100% accurate
- For production use, consider implementing true diarization with pyannote.audio

---

**Happy Transcribing! 🎉**

For questions or issues, ensure all dependencies are properly installed and ffmpeg is accessible from your terminal.
