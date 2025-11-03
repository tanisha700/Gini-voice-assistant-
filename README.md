# Gini_Voice_Assistant :
An intelligent, real-time voice and vision assistant built with Python. Gini (Tanisha) chats naturally, recognizes faces, searches the web, tells news, plays music, jokes, checks weather, sends WhatsApp messages, and more — all offline, no API keys needed. 

 Installation (for your setup)
In Terminal:
brew install portaudio ffmpeg
pip3 install opencv-python numpy pyttsx3 SpeechRecognition pyaudio pywhatkit wikipedia pyjokes playsound requests beautifulsoup4 lxml feedparser psutil

 Run it :
cd ~/Desktop/Gini_Voice_Assistant
python3 super_assistant_memory.py

 Usage :
1]Say: “Hey Gini” — to wake it up
2]Say: “What’s the weather today?” — for weather updates
3]Say: “Tell me a joke” — to hear something funny
4]Say: “Enroll new face” — to add yourself or someone new
5]Say: “Play music” — for instant music playback
6]Say: “Send WhatsApp message” — to send a voice-commanded message
7]Say: “What’s the latest news?” — for live headlines
8]say: "Remember i love to watching Movies" - for memory 

Key Features :
1] Real-Time Voice Chat — Talk naturally without pausing; Gini listens and replies instantly.
2] Camera Vision AI — Recognizes faces and greets users by name using OpenCV + LBPH.
3] Face Enrollment — Add a new user just by looking at the camera and saying “enroll new face”.
4] Female Voice Personality — Smooth and natural female voice for friendly, expressive replies.
5] Web Search Integration — Ask anything; it finds answers directly from the internet.
6] Weather Reports — Get your current location’s weather instantly.
7] Live News Updates — Hear the latest news headlines in real time.
8] Play Music — Say “play music” to start your favorite songs.
9] Jokes & Fun — Lighten your mood with AI-powered humor.
10] Send WhatsApp Messages — Send messages or open WhatsApp by voice command.
11]Email Assistant — Compose and send emails with just your voice.
12]No API Keys Needed — Works fully offline for most features.
13]Memory reminder

Tech Stack :
1]Python 3.13.7 (Homebrew)
2]OpenCV — Real-time face recognition
3]SpeechRecognition — Voice input
4]pyttsx3 — Text-to-speech (female voice)
5]pywhatkit — WhatsApp & web automation
6]Wikipedia + pyjokes + feedparser — Knowledge, humor, and live news
7]Flask (optional) — For web UI or camera streaming

 Vision System :
1]Uses OpenCV’s Haar Cascade and LBPH Face Recognizer to:
2]Detect faces in real time
3]Recognize known users
4]Ask permission to enroll new faces
5]Greet users personally when detected
