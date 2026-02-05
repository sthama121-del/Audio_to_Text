import streamlit as st
import whisper
import torch
import tempfile
import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

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
        
        # old code
        # Save preprocessed audio to temporary file
        #preprocessed_path = audio_path.replace('.', '_clean.')
        #sf.write(preprocessed_path, audio_normalized, sr)

        # New code - creates a .wav file instead
        preprocessed_path = audio_path.rsplit('.', 1)[0] + '_clean.wav'  # tmpXXX.m4a -> tmpXXX_clean.wav
        sf.write(preprocessed_path, audio_normalized, sr)  # ✅ soundfile can write WAV
        
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
        formatted_text += f"[{timestamp}] {text}\n\n"
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
        formatted_text += f"[{timestamp}] {speaker_label}: {text}\n\n"
    
    return formatted_text

def main():
    st.title("🎙️ Audio Transcription & Speaker Diarization App")
    st.markdown("""
    Upload an audio file (podcast, interview, etc.) and get a text transcription with optional speaker identification.
    """)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_size = st.selectbox(
            "Whisper Model Size",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,  # Default to "base"
            help="Larger models are more accurate but slower. 'base' or 'small' recommended for most uses."
        )
        
        st.markdown("---")
        
        format_option = st.radio(
            "Output Format",
            options=[
                "Plain Text",
                "With Timestamps",
                "Simple Speaker Detection"
            ],
            help="Simple Speaker Detection uses pause patterns (not as accurate as true diarization)"
        )
        
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
    
    # Main content area
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
            
            # Transcribe button
            if st.button("🎯 Transcribe Audio", type="primary", use_container_width=True):
                clean_audio_path = None
                tmp_file_path = None
                
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Validate and preprocess audio
                    with st.spinner("🔄 Preprocessing audio..."):
                        clean_audio_path, error = validate_and_preprocess_audio(tmp_file_path)
                        
                        if error:
                            st.error(f"❌ {error}")
                            st.info("💡 **Tips:**\n- Ensure the audio file contains speech\n- Try a different audio file\n- Check that the file isn't corrupted")
                            return
                    
                    # Load model
                    with st.spinner(f"Loading Whisper '{model_size}' model..."):
                        model = load_whisper_model(model_size)
                    
                    # Transcribe
                    with st.spinner("Transcribing audio... This may take a few minutes."):
                        result = transcribe_audio(clean_audio_path, model)
                    
                    if result is None:
                        st.error("❌ Transcription failed. Please try a different audio file.")
                        return
                    
                    # Check if transcription is empty
                    if not result.get('text') or not result.get('text').strip():
                        st.warning("⚠️ No speech detected in the audio file.")
                        return
                    
                    # Format output based on user selection
                    if format_option == "Plain Text":
                        transcription = result['text']
                    elif format_option == "With Timestamps":
                        transcription = format_transcription_with_timestamps(result)
                    else:  # Simple Speaker Detection
                        transcription = simple_speaker_detection(result, num_speakers)
                    
                    # Store in session state
                    st.session_state.transcription = transcription
                    
                    st.success("✅ Transcription completed!")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.exception(e)
                
                finally:
                    # Clean up temp files
                    for path in [tmp_file_path, clean_audio_path]:
                        if path and os.path.exists(path):
                            try:
                                os.unlink(path)
                            except:
                                pass
    
    with col2:
        st.header("📝 Transcription Result")
        
        if st.session_state.transcription:
            # Display transcription
            st.text_area(
                "Transcription",
                value=st.session_state.transcription,
                height=400,
                help="Your transcribed text appears here"
            )
            
            # Download button
            st.download_button(
                label="⬇️ Download Transcript (.txt)",
                data=st.session_state.transcription,
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Statistics
            word_count = len(st.session_state.transcription.split())
            char_count = len(st.session_state.transcription)
            
            st.markdown("---")
            st.markdown(f"**Statistics:**")
            st.markdown(f"- Words: {word_count:,}")
            st.markdown(f"- Characters: {char_count:,}")
        else:
            st.info("👈 Upload an audio file and click 'Transcribe Audio' to begin")
    
    # Footer with instructions
    st.markdown("---")
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        ### Instructions:
        
        1. **Upload**: Click the file uploader and select your audio file (MP3, WAV, M4A, etc.)
        2. **Configure**: Choose your preferred model size and output format in the sidebar
        3. **Transcribe**: Click the "Transcribe Audio" button and wait for processing
        4. **Download**: Once complete, you can download the transcript as a .txt file
        
        ### Tips:
        
        - **Model Selection**: Use 'base' or 'small' for faster results with good accuracy
        - **Speaker Detection**: The simple speaker detection uses pause patterns to guess when speakers change
        - **Long Files**: Larger audio files will take longer to process
        - **Accuracy**: For better speaker diarization, consider using pyannote.audio (see advanced setup)
        - **Audio Quality**: Clear audio with minimal background noise produces better results
        """)
    
    with st.expander("🔧 Advanced: True Speaker Diarization"):
        st.markdown("""
        ### For More Accurate Speaker Identification:
        
        To get true speaker diarization (accurate speaker identification), you need to install `pyannote.audio`:
```bash
        pip install pyannote.audio
```
        
        You'll also need a Hugging Face token to access the diarization models:
        
        1. Create a free account at https://huggingface.co
        2. Accept the user agreement for the model at: https://huggingface.co/pyannote/speaker-diarization
        3. Create an access token at: https://huggingface.co/settings/tokens
        
        **Note**: The current app uses a simple heuristic method that's less accurate but doesn't require additional setup.
        """)

if __name__ == "__main__":
    main()