from transformers import AutoModelForAudioClassification
import librosa, torch
from pydub import AudioSegment
import io
import time
import warnings


def convert_audio_to_wav(audio_path):
    """Convert non-WAV audio files (e.g., .m4a) to a WAV format in-memory for librosa processing."""
    if audio_path.endswith(".wav"):
        return audio_path  # Already WAV, no need to convert

    # Load audio and convert to WAV
    audio = AudioSegment.from_file(audio_path)
    
    # Use an in-memory buffer instead of saving to a file
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    
    # Reset buffer position for reading
    wav_buffer.seek(0)
    
    return wav_buffer  # Now this is an in-memory WAV file


def emo_predictor(audio_path):
    # Load model
    model = AutoModelForAudioClassification.from_pretrained("3loi/SER-Odyssey-Baseline-WavLM-Categorical-Attributes", trust_remote_code=True)

    # Get mean/std
    mean = model.config.mean
    std = model.config.std
    
    # Check if the audio is already in WAV format
    audio_path = convert_audio_to_wav(audio_path)
    
    # Load the audio into librosa
    raw_wav, _ = librosa.load(audio_path, sr=model.config.sampling_rate)
    
    # Normalize the audio by mean/std
    norm_wav = (raw_wav - mean) / (std+0.000001)
    
    # Generate the mask
    mask = torch.ones(1, len(norm_wav))
    
    # Batch it (add dim)
    wavs = torch.tensor(norm_wav).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        pred = model(wavs, mask)
        
    # Convert logits to probability
    probabilities = torch.nn.functional.softmax(pred, dim=1).squeeze().numpy()
       
    # Convert to dictionary format
    id2label = model.config.id2label
    emotion_probs = {id2label[i]: probabilities[i] for i in range(len(probabilities))}

    return emotion_probs

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Count the seconds
    start_time = time.time()
    emotion_probs = emo_predictor("m4atestfolder/suprise.m4a")
    
    # Display results
    print("Time taken:", time.time() - start_time)
    print("Emotion Probabilities:", emotion_probs)
    