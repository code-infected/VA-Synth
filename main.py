import streamlit as st
import openai
import wave
from google.cloud import speech, texttospeech
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if any)
load_dotenv()

st.title("AI-Powered Video Audio Replacement")

# Input fields for Azure API Key and URL
azure_api_key = st.text_input("Enter your Azure API Key", type="password")
azure_api_url = st.text_input("Enter your Azure API URL")

def extract_audio_from_video(video_path):
    """Extracts audio from a video file and returns the path to the audio file."""
    try:
        video = VideoFileClip(video_path)
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        audio = video.audio
        audio.write_audiofile(temp_audio_path, codec='pcm_s16le')  # Use WAV format
        return temp_audio_path
    except Exception as e:
        st.error(f"Error during audio extraction: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribes audio using Google Speech-to-Text API."""
    try:
        # Open the audio file using wave to read its properties
        with wave.open(audio_path, "rb") as audio_file:
            sample_rate = audio_file.getframerate()

        client = speech.SpeechClient()
        
        # Read the audio file content for transcription
        with open(audio_path, "rb") as audio_file_content:
            content = audio_file_content.read()
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,  # Use the detected sample rate
            language_code="en-US"
        )
        
        # Perform transcription
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
            "messages": [{"role": "user", "content": f"Please correct this transcript: {transcription}"}],
            "max_tokens": 1000
        }
        response = requests.post(azure_api_url, headers=headers, json=data)
        response_data = response.json()
        corrected_text = response_data['choices'][0]['message']['content']
        return corrected_text
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
            name="en-US-JourneyNeural"  # Adjust this to use the desired voice
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
