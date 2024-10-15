import streamlit as st
import openai
from google.cloud import speech, texttospeech
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import requests
import os
from dotenv import load_dotenv

# Load environment variables from the .env file for secure access to keys
load_dotenv()

# If keys are set in the .env file, they will be used as default values
default_api_key = os.getenv("AZURE_API_KEY", "")
default_api_url = os.getenv("AZURE_API_URL", "")

# App title and description
st.title("AI-Powered Video Audio Replacement")
st.write("This tool allows you to replace the audio in a video file with a cleaner, AI-generated voiceover.")

# Input fields for Azure OpenAI API Key and URL
# These allow users to securely input their own keys, or defaults are taken from environment variables
azure_api_key = st.text_input("Enter Azure API Key", default_api_key, type="password")
azure_api_url = st.text_input("Enter Azure API URL", default_api_url)

def transcribe_audio(audio_path):
    """Transcribe the audio from a video file using Google Speech-to-Text."""
    try:
        client = speech.SpeechClient()
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(language_code="en-US")
        response = client.recognize(config=config, audio=audio)
        
        # Join all transcription parts into a single string
        transcription = " ".join([result.alternatives[0].transcript for result in response.results])
        return transcription
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

def correct_transcription(transcription):
    """Use Azure OpenAI (GPT) to clean up the transcription, removing filler words and correcting grammar."""
    try:
        headers = {"Content-Type": "application/json", "api-key": azure_api_key}
        data = {
            "messages": [{"role": "user", "content": f"Please correct this transcript: {transcription}"}],
            "max_tokens": 1000
        }
        response = requests.post(azure_api_url, headers=headers, json=data)
        response_data = response.json()
        
        # Extract the corrected transcription
        corrected_text = response_data['choices'][0]['message']['content']
        return corrected_text
    except Exception as e:
        st.error(f"Error during transcription correction: {e}")
        return None

def synthesize_speech(text):
    """Generate AI voiceover using Google Text-to-Speech with a natural-sounding voice."""
    try:
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", 
            name="en-US-JourneyNeural"  # Choosing a lifelike voice
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        
        # Generate audio from text
        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.error(f"Error during text-to-speech synthesis: {e}")
        return None

def replace_audio_in_video(video_path, new_audio_content):
    """Replace the audio in the video file with the new AI-generated voiceover."""
    try:
        video = VideoFileClip(video_path)
        
        # Write the new audio to a temporary file
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        with open(temp_audio_path, "wb") as f:
            f.write(new_audio_content)
        
        # Replace video audio with the new one
        audio = AudioFileClip(temp_audio_path)
        video_with_new_audio = video.set_audio(audio)
        
        # Save the modified video to a temporary file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        video_with_new_audio.write_videofile(output_path, codec="libx264")
        
        # Clean up temporary files
        os.remove(temp_audio_path)
        return output_path
    except Exception as e:
        st.error(f"Error during video processing: {e}")
        return None

# Interface for uploading video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
if uploaded_file and azure_api_key and azure_api_url:
    # Save the uploaded video file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_path = temp_video.name
    
    with st.spinner("Transcribing audio..."):
        transcription = transcribe_audio(temp_path)
    
    if transcription:
        with st.spinner("Correcting transcription..."):
            corrected_transcription = correct_transcription(transcription)
        
        if corrected_transcription:
            with st.spinner("Generating new audio..."):
                new_audio = synthesize_speech(corrected_transcription)
            
            if new_audio:
                with st.spinner("Replacing audio in video..."):
                    output_video_path = replace_audio_in_video(temp_path, new_audio)
                
                if output_video_path:
                    st.success("Audio replaced successfully! Download your video below.")
                    with open(output_video_path, "rb") as video_file:
                        st.download_button(label="Download Video", data=video_file, file_name="output.mp4")
                    os.remove(output_video_path)
        os.remove(temp_path)
else:
    st.warning("Please provide a video file and complete the API details.")
