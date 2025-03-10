import whisper
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from scipy.special import softmax

# The different emotion categories
EMOTIONS = {0: 'Angry', 1: 'Sad', 2: 'Happy', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Contempt', 7: 'Neutral'}

# Load models
# Whisper model for speech recognition
whisper_model = whisper.load_model("small")
# Load Wav2Vec2 emotion model (we can change this to some other model bc this does not predict very well)
emotion_model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(emotion_model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(emotion_model_name)
model.eval()


def transcribe_audio(audio_path):
    # Convert speech to text using Whisper.
    return whisper_model.transcribe(audio_path)["text"]


def predict_emotion(audio_path):
    # Predict emotions from audio file.
    # Load audio
    waveform, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Convert to probabilities
    probabilities = softmax(logits.numpy().flatten())
    return probabilities.tolist()


def process_audio(audio_path):
    # Process audio file: speech-to-text + emotion recognition
    text = transcribe_audio(audio_path)
    # emotion_probs = predict_emotion(audio_path)
    # emotion_dict = {EMOTIONS[i]: prob for i, prob in enumerate(emotion_probs)}
    return {"text": text, "emotions": None}




if __name__ == "__main__":
    print("Setting up")

    # Happy
    audio_file = "data/audio_examples/happy_text.m4a"  # Replace with your file
    result = process_audio(audio_file)
    print(result)

    # Sad
    audio_file_sad = "data/audio_examples/sad_text.m4a"  # Replace with your file
    result_sad = process_audio(audio_file)
    print(result_sad)
