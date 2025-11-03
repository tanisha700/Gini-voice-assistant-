
brew install cmake libomp ffmpeg portaudio
python3.13 -m venv gini-venv
source gini-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# ---------
# PyTorch (Apple MPS)
# ---------
# IMPORTANT: Use official command from https://pytorch.org/get-started/locally/
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ---------
# Mediapipe note
# ---------
# If mediapipe fails, try:
# pip uninstall mediapipe
# pip install mediapipe-silicon
