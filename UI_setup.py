import sounddevice as sd
import numpy as np
import wave
import tempfile
import tkinter as tk
from tkinter import Label, Button

from text_detection import process_audio

recording = []
is_recording = False
fs = 16000  # Sampling rate
recording_stream = None  # Handle for the recording stream

# Style Dictionary
STYLE = {
    "bg_color": "#f4f4f4",
    "title_font": ("Arial", 24, "bold"),
    "button_font": ("Arial", 16),
    "button_width": 20,
    "button_height": 2,
    "button_start_color": "#4CAF50",
    "button_text_color": "white",
    "transcription_font": ("Arial", 18),
    "exit_button_color": "gray",
    "exit_button_font": ("Arial", 14),
}


def callback(indata, frames, time, status):
    """ Callback function to store recorded audio """
    if is_recording:
        recording.append(indata.copy())


def start_recording(event):
    """ Starts recording when button is pressed """
    global is_recording, recording, recording_stream
    recording = []
    is_recording = True

    # Start audio stream
    recording_stream = sd.InputStream(callback=callback, samplerate=fs, channels=1)
    recording_stream.start()

    # Schedule auto-stop after 10 sec
    root.after(10000, stop_recording)


def stop_recording(event=None):
    """ Stops recording when button is released or after 10 seconds """
    global is_recording, recording_stream
    is_recording = False

    if recording_stream:
        recording_stream.stop()
        recording_stream.close()
        recording_stream = None

    save_and_process_audio()


def save_and_process_audio():
    """ Saves recorded audio and processes it """
    if not recording:
        return

    audio_data = np.concatenate(recording, axis=0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        with wave.open(temp_wav.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    result = process_audio(temp_wav.name)
    transcription_label.config(text=result["text"])  # Update UI


def setup_ui():
    global transcription_label, root

    root = tk.Tk()
    root.title("Voice Detection")
    root.attributes('-fullscreen', True)  # Full-screen mode
    root.configure(bg=STYLE["bg_color"])

    # Title Label
    title_label = Label(root, text="Speak with your friend", font=STYLE["title_font"], bg=STYLE["bg_color"])
    title_label.pack(pady=50)

    # Recording Button (Single button for start/stop)
    record_button = Button(
        root, text="Hold to Speak",
        font=STYLE["button_font"], bg=STYLE["button_start_color"], fg=STYLE["button_text_color"],
        width=STYLE["button_width"], height=STYLE["button_height"]
    )
    record_button.pack(pady=30)

    # Bind press and release actions
    record_button.bind("<ButtonPress>", start_recording)
    record_button.bind("<ButtonRelease>", stop_recording)

    # Transcription Label
    transcription_label = Label(root, text="Transcription will appear here...", font=STYLE["transcription_font"],
                                bg=STYLE["bg_color"], wraplength=800)
    transcription_label.pack(pady=30)

    # Close App Button
    close_button = Button(
        root, text="Exit", command=root.destroy, font=STYLE["exit_button_font"],
        bg=STYLE["exit_button_color"], fg="white", width=15
    )
    close_button.pack(side=tk.BOTTOM, pady=20)

    root.mainloop()


if __name__ == "__main__":
    setup_ui()
