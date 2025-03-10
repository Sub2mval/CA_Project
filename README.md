# Emotion Detection - CA Project

## Folder Structure
```
CA_PROJECT/
│── emorecognition/
│   │── m4atestfolder/  # Folder containing test audio files
│   │   │── stressed.m4a
│   │   │── suprise.m4a
│   │   │── test.m4a
│   │   │── test2.m4a
│   │   │── test4.m4a
│   │   │── test5.m4a
│   │── __init__.py
│   │── emreco.py  # Main emotion recognition script
│── .gitignore
│── main.py  # Entry point to run emotion detection
│── requirements.txt  # List of dependencies
```

## Install Dependencies
1. **Create & activate a virtual environment (optional but recommended)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate     # For Windows
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use `emo_predictor`
The function `emo_predictor(audio_path)` takes an audio file as input and returns a dictionary mapping emotion labels to their respective probabilities.

### Example Usage:
```python
from emorecognition.emreco import emo_predictor

audio_file = "emorecognition/m4atestfolder/suprise.m4a"
emotion_probs = emo_predictor(audio_file)

print("Emotion Probabilities:", emotion_probs)
```

🚀 Now you can use `emo_predictor` to analyze speech emotions! 🎤

