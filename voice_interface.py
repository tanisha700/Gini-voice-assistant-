# voice_interface.py
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import subprocess
import time

def speak_text(text, lang='en'):
    """
    Convert text to speech with gTTS and play using system audio (afplay on macOS).
    This avoids playsound issues on macOS.
    """
    if not text:
        return
    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmpfile = f.name
        tts.write_to_fp(f)
    try:
        subprocess.call(["afplay", tmpfile])
    except Exception:
        try:
            subprocess.call(["mpg123", "-q", tmpfile])
        except Exception:
            print("Cannot play audio. Please install afplay/mpg123.")
    try:
        os.remove(tmpfile)
    except Exception:
        pass

def listen_for_speech(timeout=5, phrase_time_limit=12):
    """
    Listens from the default microphone and returns recognized text.
    Uses SpeechRecognition (Google Web Speech API). No API key needed.
    timeout: number of seconds to wait for phrase to start (None = indefinite)
    phrase_time_limit: max seconds for each phrase (avoid very long recordings).
    """
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("Listening... speak now.")
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        text = r.recognize_google(audio)
        print("User said:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Recognition error: {e}")
        return ""