import sounddevice as sd
import numpy as np
import wave
import tempfile
import tkinter as tk
from tkinter import ttk
import json
from pathlib import Path
import os
import time

from textrecongnition.text_detection import process_audio
from emorecognition.emreco import emo_predictor
from chains.main import conversational_rag_chain
# from chains.main import store_init, store_messages_on_exit
# from chains.main import store_init_2, store_messages_on_exit_2
from textrecongnition.text_to_speech import text_to_speech
from langgraph.checkpoint.memory import MemorySaver

recording = []
full_conversation = []
is_recording = False
fs = 16000  # Sampling rate
recording_stream = None
memory = MemorySaver()
DATA_DIR = Path("data/user_data")
os.makedirs(DATA_DIR, exist_ok=True)

def callback(indata, frames, time, status):
    """ Callback function to store recorded audio """
    if is_recording:
        recording.append(indata.copy())

# def save_conversation_history(user_id, conversation, data_dir):
#     user_file = data_dir / f"{user_id}.json"
#     try:
#         with open(user_file, "w", encoding="utf-8") as f:
#             json.dump(conversation, f, indent=2, ensure_ascii=False)
#         print(f"Conversation saved for user: {user_id}")
#     except Exception as e:
#         print(f"Failed to save conversation for {user_id}: {e}")


# def load_initial_messages(user_id: str) -> list:
#     """Load previous messages for a user at the start of a conversation"""
#     user_file = DATA_DIR / f"{user_id}.json"
#     if user_file.exists():
#         try:
#             with open(user_file, 'r') as f:
#                 # Load previous messages
#                 print(f"Loading initial messages for {user_id}...")
#                 return json.load(f)
#         except (IOError, json.JSONDecodeError) as e:
#             print(f"Error loading initial messages for {user_id}: {e}")
#             return []

def load_initial_messages_as_string(user_id: str) -> str:
    """Load previous messages and convert them to a plain string conversation log."""
    user_file = DATA_DIR / f"{user_id}.json"
    if user_file.exists():
        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                messages = json.load(f)
                print(f"Loaded {len(messages)} messages for user: {user_id}")

                conversation_str = ""
                for msg in messages:
                    speaker = msg.get("type", "unknown")
                    content = msg.get("content", "")
                    conversation_str += f"{speaker}: {content.strip()}\n\n"

                return conversation_str.strip()

        except Exception as e:
            print(f"❌ Error loading messages for {user_id}: {e}")
    return ""


def start_recording(event, record_button=None, stop_button = None):
    """ Starts recording when button is pressed """
    # Change Start button appearance
    if record_button:
        record_button.config(relief=tk.SUNKEN, bg="#00804C", fg="#001F3F")  # Pressed color
    if stop_button:
        stop_button.config(relief=tk.SUNKEN, bg="#1E488F", fg="#001F3F")
    global is_recording, recording, recording_stream
    recording = []
    is_recording = True
    recording_stream = sd.InputStream(callback=callback, samplerate=fs, channels=1)
    recording_stream.start()


def stop_recording(event=None, record_button=None, stop_button=None):
    """ Stops recording when button is released """

    # Reset Start button appearance
    if record_button:
        record_button.config(relief=tk.RAISED, bg="#74C365")  # Original greenish-teal

    if stop_button:
        stop_button.config(relief=tk.RAISED, bg="#001F3F")  # Stop button color

    global is_recording, recording_stream
    is_recording = False

    if recording_stream:
        recording_stream.stop()
        recording_stream.close()
        recording_stream = None

    save_and_process_audio()


# def chain_response(text_result, history = ''):
#     return conversational_rag_chain({"context": text_result["emotions"], "input": text_result["text"]}, 1)
def chain_response(text_result, history_context=""):
    combined_input = f"{history_context.strip()}\n\n{text_result['text']}".strip()
    return conversational_rag_chain({
        "context": text_result["emotions"],
        "input": combined_input
    }, 1)



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
    add_message(result["text"], "right")  # User message
    response = chain_response(result)
    add_message(response, "left")  # Bot response

    # Speak out the AI response
    text_to_speech(response)  # Calls the TTS function


def save_conversation_history(user_id, new_conversation, data_dir):
    user_file = data_dir / f"{user_id}.json"
    all_messages = []

    if user_file.exists():
        try:
            with open(user_file, "r", encoding="utf-8") as f:
                all_messages = json.load(f)
        except json.JSONDecodeError:
            pass  # Continue with empty history

    all_messages.extend(new_conversation)

    try:
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(all_messages, f, indent=2, ensure_ascii=False)
        print(f"Conversation appended for user: {user_id}")
    except Exception as e:
        print(f"Failed to save conversation for {user_id}: {e}")


def add_message(text, side):
    """ Adds a new message bubble to the conversation """
    bg_color = "#74C365" if side == "left" else "#001F3F"
    fg_color = "#001F3F" if side == "left" else "#F6F7ED"

    bubble_frame = tk.Frame(chat_frame, bg=bg_color, padx=10, pady=5)
    bubble_label = tk.Label(
        bubble_frame, text=text, wraplength=400,
        fg=fg_color,
        bg=bg_color, font=("Arial", 14)
    )
    bubble_label.pack()
    bubble_frame.pack(anchor="e" if side == "right" else "w", pady=5, padx=10)

    chat_canvas.update_idletasks()
    chat_canvas.yview_moveto(1)  # Auto-scroll
    
    full_conversation.append({
        "type": "human" if side == "right" else "ai",
        "content": text
    })

def setup_ui():
    global root, chat_frame, chat_canvas,current_user_id, initial_messages

    root = tk.Tk()
    root.title("Voice Chat")
    root.attributes('-fullscreen', True)
    root.configure(bg="#F6F7ED")  # Background color

    login_frame = tk.Frame(root)
    login_frame.pack(pady=50)
    tk.Label(login_frame, text="Enter your username (eg, can be your student ID):", font=("Arial", 16)).pack()
    user_entry = tk.Entry(login_frame, font=("Arial", 16))
    user_entry.pack(pady=10)

    # def start_session():
    #     global current_user_id
    #     current_user_id = user_entry.get()
    #     # if not current_user_id:
    #     #     current_user_id = "guest_" + str(int(time.time()))
    #     print(f"Starting session for user: {current_user_id}")
    #     initial_messages = load_initial_messages_as_string(current_user_id)
    #     # store_init_2(current_user_id, initial_messages,1)
    #     login_frame.destroy()
    
    
    #     create_chat_interface()
    def start_session():
        global current_user_id, initial_messages
        current_user_id = user_entry.get()
        print(f"Starting session for user: {current_user_id}")
        
        # Load history as string
        initial_messages = load_initial_messages_as_string(current_user_id)
        
        login_frame.destroy()
        create_chat_interface()

        # 👇 Inject initial greeting with history as RAG context
        greeting_input = {"text": "Here you have the conversational histroy of student, if you get no histroy, generate greeting", "emotions": "neutral"}
        greeting_response = chain_response(greeting_input, history_context=initial_messages)
        # add_message("Hello!", "right")  # Display the user's greeting
        add_message(greeting_response, "left")  # Display the bot's first response
        text_to_speech(greeting_response)

    tk.Button(login_frame, text="Start Session", command=start_session,
              font=("Arial", 14), bg="#4CAF50", fg="white").pack()

    root.mainloop()
    #root.mainloop()

def on_exit():
    # store_messages_on_exit(current_user_id, DATA_DIR,1)
    save_conversation_history(current_user_id, full_conversation, DATA_DIR)
    root.destroy()

def create_chat_interface():
    # Title Label
    global chat_frame,chat_canvas
    title_label = tk.Label(root, text="How are you doing?", font=("Arial", 24, "bold"), fg="#00804C", bg="#F6F7ED")
    title_label.pack(pady=20)

    # Conversation area (Centered, 60% width)
    chat_container = tk.Frame(root, width=int(root.winfo_screenwidth() * 0.8), bg="#EFEFEF")
    chat_container.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

    chat_canvas = tk.Canvas(chat_container, bg="#F6F7ED")
    scrollbar = ttk.Scrollbar(chat_container, orient="vertical", command=chat_canvas.yview)
    chat_frame = tk.Frame(chat_canvas, bg="#F6F7ED")
    chat_frame.bind("<Configure>", lambda e: chat_canvas.configure(scrollregion=chat_canvas.bbox("all")))

    chat_canvas.create_window((0, 0), window=chat_frame, anchor="nw", width=int(root.winfo_screenwidth() * 0.6))
    chat_canvas.configure(yscrollcommand=scrollbar.set)

    chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Button frame for Start & Stop Recording (aligned horizontally)
    button_frame = tk.Frame(root, bg="#F6F7ED")
    button_frame.pack(pady=20)

    record_button = tk.Button(button_frame, text="Start Recording", font=("Arial", 16), bg="#74C365", fg="#001F3F",
                              width=15, height=2)
    record_button.grid(row=0, column=0, padx=10)

    stop_button = tk.Button(button_frame, text="Stop Recording", font=("Arial", 16), bg="#1E488F", fg="#F6F7ED",
                            width=15, height=2)
    stop_button.grid(row=0, column=1, padx=10)
    stop_button.bind("<ButtonRelease>", lambda event: stop_recording(event, record_button, stop_button))
    record_button.bind("<ButtonPress>", lambda event: start_recording(event, record_button, stop_button))
    # Exit Button (Centered below)
    exit_button = tk.Button(root, text="Exit", command=on_exit, font=("Arial", 14), bg="gray", fg="white",
                            width=15)
    exit_button.pack(pady=20)



if __name__ == "__main__":
    setup_ui()
