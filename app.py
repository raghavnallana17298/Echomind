import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import ollama
import speech_recognition as sr
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from gtts import gTTS  # Import gTTS for text-to-speech
import os

# Load the trained emotion detection model
model = load_model('cnn_audio_model.h5')

# Define the class names and their corresponding indices
class_names = {
    0: 'sadness',
    1: 'fear',
    2: 'disgust',
    3: 'joy',
    4: 'surprise',
    5: 'neutral',
    6: 'anger'
}

# Load the label encoder classes
try:
    label_encoder_classes = np.load('label_encoder_classes.npy')
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_classes
except FileNotFoundError:
    st.error("Label encoder file not found. Make sure 'label_encoder_classes.npy' is present.")
    st.stop()

# Function to extract audio features using Librosa
def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)  # Taking the mean across time
        return mfccs_processed
    except Exception as e:
        st.error(f"Error extracting features from {audio_path}: {str(e)}")
        return None

# Function to recognize speech using SpeechRecognition
def recognize_speech(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)  # Convert speech to text using Google API
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Error connecting to the speech recognition service."

# Function to generate speech output from text using gTTS
def text_to_speech(text, filename="response.mp3"):
    #tts = gTTS(text=text, lang='en') # tts = gTTS('hello', lang='en', tld='com.au') # co.in, # slow=True
    tts = gTTS(text=text, lang='en', tld='com.hk')
    #tts = gTTS(text=text, lang='en', tld='co.in') # tts = gTTS('hello', lang='en', tld='com.au') # co.in, # slow=True
    tts.save(filename)
    return filename

# Function to interact with Ollama (Llama3 model) - Informal tone
def get_ollama_response(prompt, emotion):
    response = ollama.chat(
        model="tinyllama",
        messages=[{"role": "user", "content": f"{emotion.capitalize()} mood, huh? Got it. {prompt}"}],
        options={"max_tokens": 30}  # Adjusted token limit for better response
    )
    return response['message']['content']


# Function to generate chatbot response based on emotion
def generate_chatbot_response(chat_history, emotion, user_input):
    emotion_responses = {
        'sadness': "Hi there! I can sense things might feel a bit heavy today. I’m here for you. What’s on your mind?",
        'fear': "It’s okay to feel scared sometimes. I’m here to help you feel safe. Can you share what’s worrying you?",
        'disgust': "I hear you. Dealing with things that feel wrong or upsetting can be hard. Want to talk more about it?",
        'joy': "That’s fantastic! I’m so happy to hear this. What’s making you feel this way?",
        'surprise': "Wow! That’s interesting. Tell me more about what happened!",
        'neutral': "I’m here to chat about anything you like. What’s on your mind?",
        'anger': "I can tell you’re upset, and that’s completely valid. Let’s talk it out—what’s been bothering you?"
    }

    if not chat_history:
        return emotion_responses.get(emotion, "Hi! How can I help you today?")

    return get_ollama_response(user_input, emotion)

# Streamlit app
st.title("Emotion-Based Chatbot with Speech Output")

# Audio input method selection
input_method = st.radio("Choose an input method:", ("Upload Audio", "Use Microphone"))
import shutil 
# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process uploaded file
if input_method == "Upload Audio":
    uploaded_file = st.file_uploader("Upload an audio file for emotion detection", type=["wav"])
    #print('uploaded file is ',uploaded_file.name )
    
    if uploaded_file is not None:
        features = extract_audio_features(uploaded_file)
        
        if features is not None:
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=-1)
            
            predictions = model.predict(features)
            predicted_class = np.argmax(predictions, axis=1)[0]
            detected_emotion = class_names.get(predicted_class, "Unknown")
            
            st.write(f"**Detected Emotion:** {detected_emotion}")

            # Recognize speech from uploaded audio
            recognized_text = recognize_speech(uploaded_file.name)
            st.write(f"**Recognized Speech:** {recognized_text}")

            if recognized_text:
                chatbot_response = generate_chatbot_response(st.session_state.chat_history, detected_emotion, recognized_text)
                st.session_state.chat_history.append({"user": recognized_text, "bot": chatbot_response})

                st.write("### Chat History")
                for entry in st.session_state.chat_history:
                    #st.write(f"**You:** {entry['user']}")
                    st.write(f"**Chatbot:** {entry['bot']}")

                # Generate and play speech response
                audio_file = text_to_speech(chatbot_response)
                st.audio(audio_file, format="audio/mp3")
        else:
            st.write("Could not extract features from the audio file.")

# Process microphone input
elif input_method == "Use Microphone":
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Recording... Please speak")
        try:
            audio_data = recognizer.listen(source, timeout=10)
            st.write("Recording finished. Processing...")
            recognized_text = recognizer.recognize_google(audio_data)
            st.write(f"**Recognized Speech:** {recognized_text}")

            # Temporarily save recorded audio
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_data.get_wav_data())

            features = extract_audio_features("temp_audio.wav")

            if features is not None:
                features = np.expand_dims(features, axis=0)
                features = np.expand_dims(features, axis=-1)
                predictions = model.predict(features)
                predicted_class = np.argmax(predictions, axis=1)[0]
                detected_emotion = class_names.get(predicted_class, "Unknown")

                st.write(f"**Detected Emotion:** {detected_emotion}")

                chatbot_response = generate_chatbot_response(st.session_state.chat_history, detected_emotion, recognized_text)
                st.session_state.chat_history.append({"user": recognized_text, "bot": chatbot_response})

                st.write("### Chat History")
                for entry in st.session_state.chat_history:
                    #st.write(f"**You:** {entry['user']}")
                    st.write(f"**Chatbot:** {entry['bot']}")

                # Generate and play speech response
                audio_file = text_to_speech(chatbot_response)
                st.audio(audio_file, format="audio/mp3")
            else:
                st.write("Could not extract features from the audio file.")
        except sr.WaitTimeoutError:
            st.write("No speech detected. Try again.")
        except sr.UnknownValueError:
            st.write("Could not understand the audio.")
