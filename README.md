# Emotion Detection - CA Project

## Introduction

This project integrates **speech emotion recognition, text detection**, and **a user-friendly voice-based UI** to analyze emotions from audio recordings. It utilizes **OpenAI Whisper** for speech-to-text, **Wav2Vec2** for emotion detection, and **Tkinter** for an interactive user interface.

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
│── textrecongnition/
│   │── text_detection.py  # Speech-to-text processing using Whisper
│──UI_setup.py  # Tkinter-based voice chat UI
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

## How to Use `process_audio` for Text Detection

The function `process_audio(audio_path)` converts speech to text using OpenAI Whisper.

### Example Usage:

```python
from textrecongnition.text_detection import process_audio

audio_file = "data/audio_examples/happy_text.m4a"
result = process_audio(audio_file)

print("Transcribed Text:", result["text"])
```

## How to Use the Voice Chat UI

The project includes a **Tkinter-based UI** that allows users to record voice, analyze emotions, and interact with a chatbot.

### **Run the UI**:

```bash
python ui/UI_setup.py
```

### **UI Features**:

- **Start & Stop Recording**: Users can record voice and analyze emotions in real time.
- **Chat History Display**: Recognized text and detected emotions are displayed interactively.
- **Exit Button**: Easily close the UI.

🚀 Now you can use `emo_predictor` for emotion detection, `process_audio` for text detection, and the interactive UI for real-time voice chat analysis! 🎤

### For the control group we used the branch 'memory_per_user_no_memory'