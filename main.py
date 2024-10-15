import streamlit as st
import openai
import wave
from google.cloud import speech, texttospeech
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import requests
import os
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables from .env file (if any)
load_dotenv()

st.title("AI-Powered Video Audio Replacement")

# Input fields for Azure API Key and URL
azure_api_key = st.text_input("Enter your Azure API Key", type="password")
azure_api_url = st.text_input("Enter your Azure API URL")


def extract_audio_from_video(video_path):
    """Extracts audio from a video file, converts to mono, and returns the path to the audio file."""
    try:
        # Extract audio from video
        video = VideoFileClip(video_path)
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')  # Use WAV format

        # Convert to mono using pydub
        audio = AudioSegment.from_file(temp_audio_path)
        mono_audio = audio.set_channels(1)
        mono_audio.export(temp_audio_path, format="wav")
        
        return temp_audio_path
    except Exception as e:
        st.error(f"Error during audio extraction: {e}")
        return None

def compress_audio(audio_path, target_dBFS=-20.0):
    """Compresses the audio file to a target dBFS level."""
    audio = AudioSegment.from_wav(audio_path)
    
    # Reduce the volume to target dBFS level
    change_in_dBFS = target_dBFS - audio.dBFS
    compressed_audio = audio.apply_gain(change_in_dBFS)
    
    # Save to a temporary file
    temp_compressed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    compressed_audio.export(temp_compressed_path, format="wav")
    
    return temp_compressed_path

def split_audio_into_chunks(audio_path, chunk_length_ms=60000):
    """Splits audio into smaller chunks."""
    audio = AudioSegment.from_wav(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def transcribe_audio(audio_path):
    """Transcribes audio using Google Speech-to-Text API."""
    try:
        # Compress audio before transcription
        compressed_audio_path = compress_audio(audio_path)
        
        # Open the audio file using wave to read its properties
        with wave.open(compressed_audio_path, "rb") as audio_file:
            sample_rate = audio_file.getframerate()

        client = speech.SpeechClient()
        
        # Read the audio file content for transcription
        with open(compressed_audio_path, "rb") as audio_file_content:
            content = audio_file_content.read()
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,  # Use the detected sample rate
            language_code="en-US"
        )
        
        # Handle chunking if audio size exceeds limit
        if os.path.getsize(compressed_audio_path) > 10 * 1024 * 1024:  # 10 MB limit
            chunks = split_audio_into_chunks(compressed_audio_path)
            transcriptions = []
            for chunk in chunks:
                with open(chunk, "rb") as audio_file:
                    content_chunk = audio_file.read()
                response = client.recognize(config=config, audio=speech.RecognitionAudio(content=content_chunk))
                transcriptions.extend([result.alternatives[0].transcript for result in response.results])
            return " ".join(transcriptions)
        else:
            response = client.recognize(config=config, audio=audio)
            transcription = " ".join([result.alternatives[0].transcript for result in response.results])
            return transcription
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

def correct_transcription(transcription):
    """Corrects transcription using Azure OpenAI GPT-4o."""
    try:
        headers = {"Content-Type": "application/json", "api-key": azure_api_key}
        data = {
            "messages": [{"role": "user", "content": f"Please correct this transcript should not say anything other than given in transcript and dont alter the transcript: {transcription}"}],
            "max_tokens": 1000
        }
        response = requests.post(azure_api_url, headers=headers, json=data)
        
        # Check for a successful response and expected content
        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                corrected_text = response_data['choices'][0]['message']['content']
                return corrected_text
            else:
                st.error("Unexpected response format from Azure OpenAI.")
                return None
        else:
            st.error(f"Error from Azure API: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        st.error(f"Error during transcription correction: {e}")
        return None


def synthesize_speech(text):
    """Synthesizes speech from text using Google Text-to-Speech API."""
    try:
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", 
            name="en-AU-Standard-B"  # Adjust this to use the desired voice
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        
        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.error(f"Error during text-to-speech synthesis: {e}")
        return None

def replace_audio_in_video(video_path, new_audio_content):
    """Replaces the audio in the video with new audio content."""
    try:
        video = VideoFileClip(video_path)
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        with open(temp_audio_path, "wb") as f:
            f.write(new_audio_content)
        
        audio = AudioFileClip(temp_audio_path)
        video_with_new_audio = video.set_audio(audio)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        video_with_new_audio.write_videofile(output_path, codec="libx264")
        
        os.remove(temp_audio_path)
        return output_path
    except Exception as e:
        st.error(f"Error during video processing: {e}")
        return None

# Streamlit interface
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
if uploaded_file and azure_api_key and azure_api_url:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_path = temp_video.name

    # Extract audio from the video
    audio_path = extract_audio_from_video(temp_path)

    if audio_path:
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(audio_path)
        
        if transcription:
            with st.spinner("Correcting..."):
                corrected_transcription = correct_transcription(transcription)
            
            if corrected_transcription:
                with st.spinner("Synthesizing..."):
                    new_audio = synthesize_speech(corrected_transcription)
                
                if new_audio:
                    with st.spinner("Replacing audio..."):
                        output_video_path = replace_audio_in_video(temp_path, new_audio)
                    
                    if output_video_path:
                        st.success("Audio replaced successfully! Download Now")
                        with open(output_video_path, "rb") as video_file:
                            st.download_button(label="Download Video", data=video_file, file_name="output.mp4")
                        os.remove(output_video_path)
    
    os.remove(temp_path)
else:
    st.warning("Please upload a video file and enter your Azure API credentials.")
