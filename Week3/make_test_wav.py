import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Optional: pick a slower rate & different voice
engine.setProperty('rate', 160)  # words per minute
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # pick the first available voice

# Save speech to a WAV file
text = "in 3 sentences or less, tell me how my perspective would change about Italy by traveling there as opposed to staying in my home country of canada."
engine.save_to_file(text, "3history.wav")

engine.runAndWait()

print("test.wav generated.")
