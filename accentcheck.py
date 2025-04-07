from gtts import gTTS  # Import gTTS for text-to-speech

tts = gTTS(text="hello world", lang='en', tld='com.hk') # tts = gTTS('hello', lang='en', tld='com.au') # co.in, # slow=True
tts.save("hello1.mp3")


tts = gTTS(text="hello world", lang='en', tld='co.in') # tts = gTTS('hello', lang='en', tld='com.au') # co.in, # slow=True
tts.save("hello2.mp3")
