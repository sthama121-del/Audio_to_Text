import streamlit as st
import whisper
import torch
import tempfile
import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import yt_dlp

st.set_page_config(
    page_title="Audio Transcription App",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = None

@st.cache_resource
def load_whisper_model(model_size="base"):
    """Load Whisper model with caching"""
    model = whisper.load_model(model_size)
    return model

def download_youtube_audio(url):
    """Download audio from YouTube URL using yt-dlp, returns path to downloaded file"""
    try:
        tmp_dir = tempfile.mkdtemp()
        output_path = os.path.join(tmp_dir, "yt_audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "audio")

        final_path = os.path.join(tmp_dir, "yt_audio.mp3")
        if os.path.exists(final_path):
            return final_path, title, None
        else:
            return None, None, "Download completed but audio file not found"

    except Exception as e:
        return None, None, f"YouTube download error: {str(e)}"

def validate_and_preprocess_audio(audio_path):
    """Validate and preprocess audio file to prevent transcription errors"""
    try:
        # Load audio with librosa (converts to 16kHz mono automatically)
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Check minimum duration
        duration = len(audio) / sr
        if duration < 0.1:
            return None, "Audio must be at least 0.1 seconds long"

        # Trim silence from beginning and end
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

        # Check again after trimming
        if len(audio_trimmed) < 1600:  # 0.1 seconds at 16kHz
            return None, "No speech detected in audio (only silence found)"

        # Normalize audio to prevent clipping
        audio_normalized = librosa.util.normalize(audio_trimmed)

        # Creates a .wav file for soundfile compatibility
        preprocessed_path = audio_path.rsplit('.', 1)[0] + '_clean.wav'
        sf.write(preprocessed_path, audio_normalized, sr)

        return preprocessed_path, None

    except Exception as e:
        return None, f"Audio preprocessing error: {str(e)}"

def transcribe_audio(audio_path, model):
    """Transcribe audio file using Whisper with robust error handling"""
    try:
        result = model.transcribe(
            audio_path,
            verbose=False,
            language=None,  # Auto-detect language
            task="transcribe",
            fp16=False,  # Use FP32 for CPU/Mac compatibility
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            temperature=0.0
        )
        return result
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def format_transcription_with_timestamps(result):
    """Format transcription with timestamps for better readability"""
    formatted_text = ""
    for segment in result['segments']:
        timestamp = format_timestamp(segment['start'])
        text = segment['text'].strip()
        formatted_text += f"• [{timestamp}] {text}\n\n"
    return formatted_text

def simple_speaker_detection(result, num_speakers=2):
    """
    Simple speaker detection based on pauses and sentence patterns.
    This is a basic heuristic approach - not as accurate as true diarization.
    """
    formatted_text = ""
    current_speaker = 0
    pause_threshold = 2.0  # seconds

    segments = result['segments']
    for i, segment in enumerate(segments):
        # Detect speaker changes based on long pauses
        if i > 0:
            pause = segment['start'] - segments[i-1]['end']
            if pause > pause_threshold:
                current_speaker = (current_speaker + 1) % num_speakers

        timestamp = format_timestamp(segment['start'])
        speaker_label = f"Speaker {chr(65 + current_speaker)}"  # A, B, C, etc.
        text = segment['text'].strip()

        formatted_text += f"• [{timestamp}] {speaker_label}:\n{text}\n\n"

    return formatted_text

def run_transcription(audio_path, model_size, format_option, num_speakers=2):
    """Shared transcription logic used by both file upload and YouTube tabs"""
    clean_audio_path = None
    try:
        with st.spinner("🔄 Preprocessing audio..."):
            clean_audio_path, error = validate_and_preprocess_audio(audio_path)
            if error:
                st.error(f"❌ {error}")
                return

        with st.spinner(f"Loading Whisper '{model_size}' model..."):
            model = load_whisper_model(model_size)

        with st.spinner("Transcribing audio... This may take a few minutes."):
            result = transcribe_audio(clean_audio_path, model)

        if result is None:
            st.error("❌ Transcription failed. Please try a different audio file.")
            return

        if not result.get('text') or not result.get('text').strip():
            st.warning("⚠️ No speech detected in the audio file.")
            return

        if format_option == "Plain Text with Bullets":
            formatted_text = ""
            for segment in result['segments']:
                text = segment['text'].strip()
                formatted_text += f"• {text}\n\n"
            transcription = formatted_text
        elif format_option == "With Timestamps":
            transcription = format_transcription_with_timestamps(result)
        else:
            transcription = simple_speaker_detection(result, num_speakers)

        st.session_state.transcription = transcription
        st.success("✅ Transcription completed!")

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

    finally:
        if clean_audio_path and os.path.exists(clean_audio_path):
            try:
                os.unlink(clean_audio_path)
            except:
                pass

def main():
    st.title("🎙️ Audio Transcription & Speaker Diarization App")
    st.markdown("Transcribe audio from a local file or directly from a YouTube URL — free, local, no API costs.")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        model_size = st.selectbox(
            "Whisper Model Size",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower. 'base' or 'small' recommended for most uses."
        )

        st.markdown("---")

        format_option = st.radio(
            "Output Format",
            options=[
                "Plain Text with Bullets",
                "With Timestamps",
                "Simple Speaker Detection"
            ],
            help="Choose how you want the transcript formatted"
        )

        num_speakers = 2
        if format_option == "Simple Speaker Detection":
            num_speakers = st.number_input(
                "Expected Number of Speakers",
                min_value=2,
                max_value=10,
                value=2,
                help="Estimate the number of speakers in the audio"
            )

        st.markdown("---")
        st.info("""
        **Model Sizes:**
        - **tiny**: Fastest, least accurate
        - **base**: Good balance (recommended)
        - **small**: More accurate, slower
        - **medium/large**: Most accurate, very slow
        """)

    # Two tabs — File Upload and YouTube URL
    tab1, tab2 = st.tabs(["📁 Upload Audio File", "▶️ YouTube URL"])

    # ── Tab 1: File Upload ──────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("📤 Upload Audio File")
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=["mp3", "wav", "m4a", "ogg", "flac"],
                help="Supported formats: MP3, WAV, M4A, OGG, FLAC"
            )

            if uploaded_file is not None:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')

                if st.button("🎯 Transcribe Audio", type="primary", use_container_width=True, key="btn_file"):
                    tmp_file_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        run_transcription(tmp_file_path, model_size, format_option, num_speakers)

                    finally:
                        if tmp_file_path and os.path.exists(tmp_file_path):
                            try:
                                os.unlink(tmp_file_path)
                            except:
                                pass

        with col2:
            st.header("📝 Transcription Result")
            if st.session_state.transcription:
                st.text_area("Transcription", value=st.session_state.transcription, height=400)
                st.download_button(
                    label="⬇️ Download Transcript (.txt)",
                    data=st.session_state.transcription,
                    file_name="transcript.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                word_count = len(st.session_state.transcription.split())
                char_count = len(st.session_state.transcription)
                st.markdown("---")
                st.markdown(f"**Statistics:** {word_count:,} words · {char_count:,} characters")
            else:
                st.info("👈 Upload an audio file and click 'Transcribe Audio' to begin")

    # ── Tab 2: YouTube URL ──────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("▶️ YouTube URL")
            youtube_url = st.text_input(
                "Paste YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Paste any YouTube video URL — audio will be downloaded and transcribed locally"
            )

            if youtube_url:
                if st.button("🎯 Download & Transcribe", type="primary", use_container_width=True, key="btn_yt"):
                    yt_audio_path = None
                    try:
                        with st.spinner("⬇️ Downloading audio from YouTube..."):
                            yt_audio_path, title, error = download_youtube_audio(youtube_url)

                        if error:
                            st.error(f"❌ {error}")
                            st.info("💡 Make sure the URL is a valid public YouTube video.")
                        else:
                            st.success(f"✅ Downloaded: {title}")
                            run_transcription(yt_audio_path, model_size, format_option, num_speakers)

                    finally:
                        if yt_audio_path and os.path.exists(yt_audio_path):
                            try:
                                os.unlink(yt_audio_path)
                            except:
                                pass

            st.markdown("---")
            st.info("""
            **How it works:**
            - Audio is downloaded locally using yt-dlp
            - No data is sent to any external API
            - Works with any public YouTube video
            - Best results with clear speech and minimal background music
            """)

        with col2:
            st.header("📝 Transcription Result")
            if st.session_state.transcription:
                st.text_area("Transcription", value=st.session_state.transcription, height=400, key="yt_result")
                st.download_button(
                    label="⬇️ Download Transcript (.txt)",
                    data=st.session_state.transcription,
                    file_name="transcript.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="yt_download"
                )
                word_count = len(st.session_state.transcription.split())
                char_count = len(st.session_state.transcription)
                st.markdown("---")
                st.markdown(f"**Statistics:** {word_count:,} words · {char_count:,} characters")
            else:
                st.info("👈 Paste a YouTube URL and click 'Download & Transcribe' to begin")

    # Footer
    st.markdown("---")
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        ### File Upload Tab
        1. Select an audio file (MP3, WAV, M4A, OGG, FLAC)
        2. Choose model size and output format in the sidebar
        3. Click Transcribe Audio
        4. Download the transcript as .txt

        ### YouTube Tab
        1. Paste any public YouTube URL
        2. Choose model size and output format in the sidebar
        3. Click Download & Transcribe
        4. Download the transcript as .txt

        ### Tips
        - Use **base** or **small** model for faster results
        - Clear audio with minimal background noise gives best results
        - Telugu, Hindi, Tamil and 90+ languages are auto-detected
        """)

    with st.expander("🔧 Advanced: True Speaker Diarization"):
        st.markdown("""
        For accurate speaker identification, install pyannote.audio:

            pip install pyannote.audio

        You will also need a free Hugging Face token from https://huggingface.co/settings/tokens
        and must accept the model agreement at https://huggingface.co/pyannote/speaker-diarization

        The current app uses a heuristic pause-based method which requires no additional setup.
        """)

if __name__ == "__main__":
    main()
