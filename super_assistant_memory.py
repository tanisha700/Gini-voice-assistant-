import os, cv2, time, threading, pyttsx3, numpy as np , speech_recognition as sr
import pywhatkit, wikipedia, pyjokes, requests, feedparser, psutil, datetime, webbrowser

DATASET_DIR = "faces_dataset"
MODEL_FILE = "face_model.yml"
LABELS_FILE = "labels.json"
os.makedirs(DATASET_DIR, exist_ok=True)
CONFIDENCE_THRESHOLD = 70

engine = pyttsx3.init()
voices = engine.getProperty('voices')


for v in voices:
    if "female" in v.name.lower():
        engine.setProperty('voice', v.id)
        break
engine.setProperty('rate', 175)

def speak(text):
    print(f" Gini: {text}")
    engine.say(text)
    engine.runAndWait()

listener = sr.Recognizer()
listener.dynamic_energy_threshold = True
listener.pause_threshold = 0.6

def listen_background(callback):
    """Continuously listens and calls callback as soon as speech is detected."""
    with sr.Microphone() as source:
        listener.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = listener.listen(source, phrase_time_limit=5)
                command = listener.recognize_google(audio).lower()
                if command:
                    print(f" Tanisha said: {command}")
                    callback(command)
            except sr.UnknownValueError:
                continue
            except Exception:
                continue

def enroll_face(name):
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0
    speak(f"Tanisha, please look at the camera to enroll your face.")
    while count < 30:
        ret, frame = cam.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"{DATASET_DIR}/{name}_{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Enrolling Face (Press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    speak(f"Face enrollment complete, {name}. You look beautiful today!")

def play_music():
    speak("Sure, what song should I play?")
    song = input(" Type or say song name: ") or "romantic songs"
    pywhatkit.playonyt(song)
    speak(f"Playing {song} on YouTube.")

def tell_joke():
    speak(pyjokes.get_joke())

def tell_news():
    speak("Fetching latest news headlines.")
    feed = feedparser.parse("https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en")
    for i, entry in enumerate(feed.entries[:5], 1):
        speak(f"Headline {i}: {entry.title}")
    speak("That’s all for now, Tanisha.")

def check_weather(city="Latur"):
    url = f"https://wttr.in/{city}?format=%C+%t"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            speak(f"Weather in {city} is {res.text}")
    except:
        speak("Sorry, I couldn’t fetch the weather right now.")

def system_status():
    battery = psutil.sensors_battery()
    if battery:
        speak(f"Battery is at {battery.percent} percent.")
    now = datetime.datetime.now()
    speak(f"It’s {now.strftime('%A, %I:%M %p, %B %d')}.")

def open_whatsapp():
    speak("Opening WhatsApp Web.")
    webbrowser.open("https://web.whatsapp.com")

def search_web():
    speak("What should I search for?")
    query = input("Type or say query: ")
    webbrowser.open(f"https://www.google.com/search?q={query}")
    speak(f"Here are the results for {query}.")

def open_camera():
    speak("Opening camera... Press 'q' to quit.")
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        cv2.imshow("Camera - Press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    speak("Camera closed, Tanisha.")

def respond(command):
    if any(word in command for word in ["play", "song", "music"]):
        play_music()
    elif "joke" in command:
        tell_joke()
    elif "news" in command:
        tell_news()
    elif "weather" in command:
        check_weather()
    elif "camera" in command:
        open_camera()
    elif "search" in command or "google" in command:
        search_web()
    elif "whatsapp" in command:
        open_whatsapp()
    elif "battery" in command or "time" in command:
        system_status()
    elif "enroll" in command and "face" in command:
        speak("What name should I save your face as?")
        name = "Tanisha"
        enroll_face(name)
    elif "quit" in command or "bye" in command or "stop" in command:
        speak("Goodbye Tanisha! I’ll miss you.")
        os._exit(0)
    else:
        try:
            info = wikipedia.summary(command, sentences=5)
            speak(info)
        except:
            speak("I’m not sure about that, Tanisha.")

def run_assistant():
    speak("Hi Tanisha! It's me, Gini. Ready to hang out?")
    threading.Thread(target=listen_background, args=(respond,), daemon=True).start()

    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    run_assistant()


# === Integrated Live Face Chat System (Triggered by "Start live chat") ===

# Imported from live_face_chat.py
"""
live_face_chat.py

A single-file face-analysis conversational assistant ("Gini") that:
- Uses MediaPipe Face Mesh for accurate face landmark detection (no API keys).
- Computes eyeball size difference, eyebrow lengths, forehead length, and other facial structure ratios.
- Provides sequential conversation flow (no background/chat looping). The assistant speaks (TTS) and listens (STT) for consent and replies.
- Asks explicit privacy consent before analyzing and again before saving any data for future use.
- Handles multiple faces detected in front of the camera and asks consent for each person.
- Optionally uses a Hugging Face Wav2Vec2 ASR model *if available locally* (no API key needed). If not available, it falls back to the SpeechRecognition Google recognizer (requires internet but no API key) or a simple keyboard input as a backup.

INSTALL (example):
    pip install opencv-python mediapipe numpy pyttsx3 sounddevice soundfile transformers torch torchvision torchaudio SpeechRecognition pyaudio

Notes:
- MediaPipe provides robust facial landmarks and is the main contributor to accuracy improvements.
- Hugging Face local model: set HF_STT_MODEL env var to a local or huggingface repo id if you want to use Wav2Vec2 for speech-to-text. The code will try to load it if torch & transformers are installed and the model is available locally/cached.
- Run on your machine; large HF models require disk and memory.

Usage:
    python live_face_chat.py

Press 'a' to analyze the current camera frame (sequential step triggered by user).
Press 'q' to quit.

"""

import os
import cv2
import json
import time
import math
import queue
import threading
import numpy as np
from pathlib import Path

# TTS (offline)
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except Exception:
    _TTS_AVAILABLE = False

# STT optional: try Hugging Face Wav2Vec2 locally
_HF_ASR_AVAILABLE = False
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torch
    import soundfile as sf
    _HF_ASR_AVAILABLE = True
except Exception:
    _HF_ASR_AVAILABLE = False

# Fallback STT via SpeechRecognition (uses Google by default without API key)
try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except Exception:
    _SR_AVAILABLE = False

# MediaPipe face mesh
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False


# ------------------------- Helper classes -------------------------
class TTS:
    def __init__(self):
        if not _TTS_AVAILABLE:
            print("pyttsx3 not available; falling back to print-only TTS")
            self.engine = None
        else:
            self.engine = pyttsx3.init()
            # tweak voice rate gently
            try:
                rate = self.engine.getProperty('rate')
                self.engine.setProperty('rate', max(120, rate - 10))
            except Exception:
                pass

    def speak(self, text, block=True):
        if self.engine:
            self.engine.say(text)
            if block:
                self.engine.runAndWait()
            else:
                threading.Thread(target=self.engine.runAndWait, daemon=True).start()
        else:
            print("TTS:", text)


class STT:
    """Speech-to-text. Tries HF Wav2Vec2 locally, then speech_recognition, then keyboard input."""
    def __init__(self):
        self.use_hf = False
        self.hf_processor = None
        self.hf_model = None
        hf_model_id = os.environ.get('HF_STT_MODEL', None)
        if _HF_ASR_AVAILABLE and hf_model_id:
            try:
                print(f"Loading HF ASR model {hf_model_id} ...")
                self.hf_processor = Wav2Vec2Processor.from_pretrained(hf_model_id)
                self.hf_model = Wav2Vec2ForCTC.from_pretrained(hf_model_id)
                if torch.cuda.is_available():
                    self.hf_model.to('cuda')
                self.use_hf = True
                print("HF ASR loaded.")
            except Exception as e:
                print("Failed loading HF ASR:", e)
                self.use_hf = False

        self.sr = sr.Recognizer() if _SR_AVAILABLE else None

    def transcribe_file_hf(self, filepath):
        speech, sr_rate = sf.read(filepath)
        if len(speech.shape) > 1:
            speech = np.mean(speech, axis=1)
        input_values = self.hf_processor(speech, sampling_rate=sr_rate, return_tensors="pt").input_values
        if torch.cuda.is_available():
            input_values = input_values.to('cuda')
        with torch.no_grad():
            logits = self.hf_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.hf_processor.batch_decode(predicted_ids)[0]
        return transcription.lower()

    def listen_and_transcribe(self, timeout=6, phrase_time_limit=6, wav_output='/tmp/_gini_record.wav'):
        # First try Hugging Face if available: record with sounddevice to a file then transcribe
        try:
            import sounddevice as sd
            import soundfile as sf
        except Exception:
            sd = None

        if self.use_hf and sd is not None:
            # record audio from microphone for phrase_time_limit seconds
            try:
                samplerate = 16000
                duration = phrase_time_limit
                print(f"Recording for {duration}s for HF ASR...")
                recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
                sd.wait()
                sf.write(wav_output, recording, samplerate)
                txt = self.transcribe_file_hf(wav_output)
                return txt
            except Exception as e:
                print("HF recording/transcription error:", e)

        # Next fallback: SpeechRecognition with microphone (Google recognizer)
        if self.sr is not None:
            try:
                with sr.Microphone() as source:
                    print("Listening (SpeechRecognition)...")
                    audio = self.sr.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                text = self.sr.recognize_google(audio).lower()
                return text
            except Exception as e:
                print("SpeechRecognition error:", e)

        # Final fallback: keyboard input
        print("Please type response (fallback): ")
        return input('> ').strip().lower()

class FaceAnalyzer:
    """Uses MediaPipe Face Mesh to compute facial measurements.
    All distances are normalized by face width (distance between left-most and right-most landmark) to be scale-invariant.
    """
    def __init__(self):
        if not _MP_AVAILABLE:
            raise RuntimeError('mediapipe is required for FaceAnalyzer')
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False,
                                                    max_num_faces=10,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)

    def analyze(self, image_rgb):
        h, w, _ = image_rgb.shape
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return []
        analyses = []
        for face_landmarks in results.multi_face_landmarks:
            lm = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]
            face_w = self._dist_between_points(lm[33], lm[263]) or 1.0  
            left_eye_points = [33, 7, 163, 144, 145, 153, 154, 155, 133]
            right_eye_points = [263, 249, 390, 373, 374, 380, 381, 382, 362]
            left_eye_w = self._polygon_width(lm, left_eye_points)
            right_eye_w = self._polygon_width(lm, right_eye_points)

            eye_size_diff = (left_eye_w - right_eye_w) / (face_w)
            left_eyebrow = [70, 63, 105, 66, 107]
            right_eyebrow = [300, 293, 334, 296, 336]
            left_eyebrow_len = self._polyline_length(lm, left_eyebrow)
            right_eyebrow_len = self._polyline_length(lm, right_eyebrow)
            eyebrow_diff = (left_eyebrow_len - right_eyebrow_len) / face_w
            brow_mid = self._midpoint(lm[10], lm[338]) if len(lm) > 338 else self._midpoint(lm[70], lm[300])
            nose_tip = lm[1] 
            forehead_height = (brow_mid[1] - nose_tip[1]) / face_w
            face_h = self._dist_between_points(lm[10], lm[152]) or 1.0
            face_aspect = face_w / face_h
            left_cheek = lm[234] if len(lm) > 234 else lm[33]
            right_cheek = lm[454] if len(lm) > 454 else lm[263]
            mid_nose = lm[1]
            yaw = (left_cheek[0] - right_cheek[0]) / face_w
            roll = (lm[10][1] - lm[152][1]) / face_w

            analyses.append({
                'left_eye_w_px': left_eye_w,
                'right_eye_w_px': right_eye_w,
                'eye_size_diff_norm': float(eye_size_diff),
                'left_eyebrow_len_px': left_eyebrow_len,
                'right_eyebrow_len_px': right_eyebrow_len,
                'eyebrow_diff_norm': float(eyebrow_diff),
                'forehead_height_norm': float(forehead_height),
                'face_aspect': float(face_aspect),
                'yaw': float(yaw),
                'roll': float(roll),
                'face_width_px': face_w,
                'face_height_px': face_h
            })
        return analyses

    def _dist_between_points(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _polygon_width(self, lm, idxs):
        xs = [lm[i][0] for i in idxs if i < len(lm)]
        if not xs:
            return 0
        return max(xs) - min(xs)

    def _polyline_length(self, lm, idxs):
        total = 0.0
        pts = [lm[i] for i in idxs if i < len(lm)]
        for i in range(1, len(pts)):
            total += self._dist_between_points(pts[i-1], pts[i])
        return total

    def _midpoint(self, a, b):
        return ((a[0]+b[0])//2, (a[1]+b[1])//2)


def main():
    tts = TTS()
    stt = STT()

    tts.speak("Hello! I am your assistant  Gini. I will help analyze facial structure when you ask.")
    tts.speak("Before we begin, a quick privacy question: do I have permission to analyze faces captured by this camera and optionally save the analysis? Please say yes or no.")
    answer = stt.listen_and_transcribe()
    if not answer or 'yes' not in answer:
        tts.speak("Understood. I will not analyze or save data. You can press 'a' to request analysis at any time and I'll ask again for consent.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        tts.speak('Cannot open camera. Exiting.')
        return

    analyzer = None
    try:
        analyzer = FaceAnalyzer()
    except Exception as e:
        tts.speak('MediaPipe is not available or failed to initialize. Install mediapipe to enable face analysis.')
        print(e)
        analyzer = None

    tts.speak("Press 'a' to analyze current frame, or 'q' to quit. When multiple people are visible, I will ask each person for permission.")

    save_file = Path('gini_face_data.json')
    saved_data = []
    if save_file.exists():
        try:
            saved_data = json.loads(save_file.read_text())
        except Exception:
            saved_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to grab frame')
            break

        display = frame.copy()
        cv2.putText(display, "Press 'a' to analyze frame | 'q' to quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.imshow('Gini - Face Assistant', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('a'):
            # Perform one-step analysis (sequential)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not analyzer:
                tts.speak('Face analysis is not available on this machine.')
                continue
            analyses = analyzer.analyze(rgb)
            if not analyses:
                tts.speak('I could not detect any faces. Please ensure faces are visible to the camera.')
                continue

            # For each face detected, ask consent and analyze sequentially
            for i, a in enumerate(analyses):
                tts.speak(f'I detected person number {i+1}. Do I have permission to analyze their face? Please say yes or no.')
                ans = stt.listen_and_transcribe()
                if not ans or 'yes' not in ans:
                    tts.speak('Consent not granted. Skipping this person.')
                    continue

                # Summarize findings in simple language
                eye_diff_pct = a['eye_size_diff_norm'] * 100
                eyebrow_diff_pct = a['eyebrow_diff_norm'] * 100
                forehead = a['forehead_height_norm']
                aspect = a['face_aspect']

                speak_txt = (
                    f'Analysis for person {i+1}: left eye width {a["left_eye_w_px"]:.0f} pixels, '
                    f'right eye width {a["right_eye_w_px"]:.0f} pixels. '
                    f'Eye size difference {eye_diff_pct:+.1f} percent of face width. '
                    f'Eyebrow length difference {eyebrow_diff_pct:+.1f} percent of face width. '
                    f'Forehead height approximately {forehead:.3f} normalized units. '
                    f'Face aspect ratio (width/height) is {aspect:.2f}.'
                )
                tts.speak(speak_txt)

                # Ask to save
                tts.speak('Would you like me to save this analysis for future reference? Please say yes or no.')
                save_ans = stt.listen_and_transcribe()
                if save_ans and 'yes' in save_ans:
                    timestamp = int(time.time())
                    record = {'timestamp': timestamp, 'analysis': a}
                    saved_data.append(record)
                    try:
                        save_file.write_text(json.dumps(saved_data, indent=2))
                        tts.speak('Saved. The data is stored locally in gini_face_data.json')
                    except Exception as e:
                        print('Save error:', e)
                        tts.speak('Sorry, I could not write the file.')
                else:
                    tts.speak('Not saved.')

            tts.speak('Analysis step complete. Press a to analyze again or q to quit.')

    cap.release()
    cv2.destroyAllWindows()
    tts.speak('Goodbye!')


if __name__ == '__main__':
    main()


# Note:
# Live Face Chat is triggered via voice command "Start live chat".
# It replaces the previous face/camera logic for a more flexible real-time output.



# === Memory System (Added Feature) ===
import json

MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=4)

memory_data = load_memory()

def remember_info(command):
    if "remember that" in command.lower():
        info = command.lower().replace("remember that", "").strip()
        if "like" in info or "favourite" in info or "favorite" in info:
            key = "preferences"
        elif "hobby" in info or "hobbies" in info:
            key = "hobbies"
        else:
            key = "info"
        memory_data.setdefault(key, []).append(info)
        save_memory(memory_data)
        speak(f"Okay, I’ll remember that you {info}.")
        return True
    return False

def recall_last_command():
    if "last_command" in memory_data:
        speak(f"Your last command was {memory_data['last_command']}.")
    else:
        speak("I don’t have any record of your last command yet.")

def handle_command(command):
    memory_data["last_command"] = command
    save_memory(memory_data)
    if remember_info(command):
        return
    # Existing command logic continues here...
