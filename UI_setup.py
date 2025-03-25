import sounddevice as sd
import numpy as np
import wave
import tempfile
import tkinter as tk
from tkinter import ttk

from textrecongnition.text_detection import process_audio
from emorecognition.emreco import emo_predictor
from chains.main import conversational_rag_chain

recording = []
is_recording = False
fs = 16000  # Sampling rate
recording_stream = None
DATA_DIR = Path("data/user_data")
os.makedirs(DATA_DIR, exist_ok=True)

def callback(indata, frames, time, status):
    """ Callback function to store recorded audio """
    if is_recording:
        recording.append(indata.copy())

def load_initial_messages(user_id: str) -> list:
    """Load previous messages for a user at the start of a conversation"""
    user_file = DATA_DIR / f"{user_id}.json"
    if user_file.exists():
        try:
            with open(user_file, 'r') as f:
                # Load previous messages
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading initial messages for {user_id}: {e}")
    return []

def start_recording(event):
    """ Starts recording when button is pressed """
    global is_recording, recording, recording_stream
    recording = []
    is_recording = True
    recording_stream = sd.InputStream(callback=callback, samplerate=fs, channels=1)
    recording_stream.start()


def stop_recording(event=None):
    """ Stops recording when button is released or after 10 seconds """
    global is_recording, recording_stream
    is_recording = False

    if recording_stream:
        recording_stream.stop()
        recording_stream.close()
        recording_stream = None

    save_and_process_audio()

def chain_response(text_result):
    return conversational_rag_chain({"context": text_result["emotions"], "input": text_result["text"],"user_id": current_user_id,"init_msg":"initial_messages": initial_messages}, 1)

#need to add a way of separating user threads

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
    #emotion_probs = emo_predictor(temp_wav.name)
    add_message(result["text"], "right")  # User message
    add_message(chain_response(result), "left")  # Bot response


def add_message(text, side):
    """ Adds a new message bubble to the conversation """
    bubble_frame = tk.Frame(chat_frame, bg='blue' if side == "right" else "green", padx=10, pady=5)
    bubble_label = tk.Label(bubble_frame, text=text, wraplength=400, fg='white' if side == "right" else "black",
                            bg='blue' if side == "right" else "green", font=("Arial", 14))
    bubble_label.pack()
    bubble_frame.pack(anchor="e" if side == "right" else "w", pady=5, padx=10)

    chat_canvas.update_idletasks()
    chat_canvas.yview_moveto(1)  # Auto-scroll


def setup_ui():
    global root, chat_frame, chat_canvas,current_user_id

    root = tk.Tk()
    root.title("Voice Chat")
    root.attributes('-fullscreen', True)

    login_frame = tk.Frame(root)
    login_frame.pack(pady=50)

    tk.Label(login_frame, text="Enter your username (eg, can be your student ID):", font=("Arial", 16)).pack()
    user_entry = tk.Entry(login_frame, font=("Arial", 16))
    user_entry.pack(pady=10)

    def start_session():
        global current_user_id
        current_user_id = user_entry.get()
        if not current_user_id:
            current_user_id = "guest_" + str(int(time.time()))
        initial_messages = load_initial_messages(current_user_id)
        login_frame.destroy()
        create_chat_interface()

    tk.Button(login_frame, text="Start Session", command=start_session,
              font=("Arial", 14), bg="#4CAF50", fg="white").pack()

    root.mainloop()

def create_chat_interface():
    # Title Label
    title_label = tk.Label(root, text="How are you doing?", font=("Arial", 24, "bold"))
    title_label.pack(pady=20)

    # Conversation area (Centered, 60% width)
    chat_container = tk.Frame(root, width=int(root.winfo_screenwidth() * 0.6))
    chat_container.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

    chat_canvas = tk.Canvas(chat_container)
    scrollbar = ttk.Scrollbar(chat_container, orient="vertical", command=chat_canvas.yview)
    chat_frame = tk.Frame(chat_canvas)

    chat_frame.bind("<Configure>", lambda e: chat_canvas.configure(scrollregion=chat_canvas.bbox("all")))

    chat_canvas.create_window((0, 0), window=chat_frame, anchor="nw", width=int(root.winfo_screenwidth() * 0.6))
    chat_canvas.configure(yscrollcommand=scrollbar.set)

    chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Button frame for Start & Stop Recording (aligned horizontally)
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)

    record_button = tk.Button(button_frame, text="Start Recording", font=("Arial", 16), bg="#4CAF50", fg="white",
                              width=15, height=2)
    record_button.grid(row=0, column=0, padx=10)
    record_button.bind("<ButtonPress>", start_recording)

    stop_button = tk.Button(button_frame, text="Stop Recording", font=("Arial", 16), bg="red", fg="white",
                            width=15, height=2)
    stop_button.grid(row=0, column=1, padx=10)
    stop_button.bind("<ButtonRelease>", stop_recording)

    # Exit Button (Centered below)
    exit_button = tk.Button(root, text="Exit", command=root.destroy, font=("Arial", 14), bg="gray", fg="white", width=15)
    exit_button.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    setup_ui()
