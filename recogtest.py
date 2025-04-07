import speech_recognition as sr
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

recognized_text = recognize_speech("Recording (3).wav")
print(f"**Recognized Speech:** {recognized_text}")
