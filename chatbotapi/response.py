import openai
import json
import datetime

# ==============================================================
# CONFIGURATION
# ==============================================================
openai.api_key = "sk-XXXX-REPLACE-WITH-YOUR-REAL-KEY"


# Specify the ChatGPT model you want to use (e.g., "gpt-3.5-turbo", "gpt-4")
CHATGPT_MODEL = "gpt-4o-mini"

# ==============================================================
# DATA STRUCTURES
# ==============================================================

# 1. Store user profile in a dictionary. Expand this as needed.
#    You can later store and load it from disk if you want persistence.
user_profile = {
    "name": "unknown",
    "major": "unknown",
    "gender": "unknown",
    "main_issues": "unknown",
    "other_notes": "unknown"
}

# 2. Conversation history for analysis (longer logs).
#    This is an ongoing list of (role, message, timestamp).
full_conversation_log = []

# 3. Short-term memory: Summaries for the current session or recent messages.
#    We'll store a short summary string that we can feed back to the model
#    as context or just hold for quick user references.
current_short_term_summary = ""


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def store_message(role, content):
    """
    Store a single message in full_conversation_log with a timestamp.
    role: "user" or "assistant"
    content: message text
    """
    msg_entry = {
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.now().isoformat()
    }
    full_conversation_log.append(msg_entry)

def summarize_short_term_history(conversation_log, max_messages=5):
    """
    Use ChatGPT to produce a short summary of the last N messages.
    Adjust `max_messages` to capture more or fewer messages.
    """
    # Gather the last N user/assistant messages
    recent_logs = conversation_log[-max_messages:]

    # Format them for a summarization prompt
    summarization_prompt = (
        "Summarize the following onversation between the user and the assistant "
        "in 2-4 sentences, capturing the key points:\n\n"
    )
    for log in recent_logs:
        summarization_prompt += f"{log['role'].upper()}: {log['content']}\n"

    # Call the ChatGPT model to generate a short summary
    response = openai.chat.completions.create(
        model=CHATGPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful summarization assistant."},
            {"role": "user", "content": summarization_prompt}
        ],
        temperature=0.7
    )
    summary = response.choices[0].message.content
    return summary

def summarize_user_profile(profile):
    """
    Summarize the user’s profile (major, name, gender, main issues, etc.)
    in a single short paragraph or bullet points.
    """
    # In practice, you might store this summary in a text file or database.
    # For demonstration, we'll just build a string:
    profile_summary = (f"Name: {profile['name']}\n"
                       f"Major: {profile['major']}\n"
                       f"Gender: {profile['gender']}\n"
                       f"Main Issues: {profile['main_issues']}\n"
                       f"Additional Notes: {profile.get('other_notes', 'N/A')}\n")
    return profile_summary


# ==============================================================
# MAIN CHATGPT FUNCTION: "Study Presser Therapist"
# ==============================================================

def chat_with_study_presser(user_input):
    """
    Takes user's input text, calls ChatGPT with the prompt to act as a
    "study presser therapist," and returns the assistant's response.
    """

    # System message can incorporate the user profile or its summary as context
    # if you want the assistant to adapt to the user’s background.
    # For example:
    system_prompt = f"""
    You are acting as a supportive study presser therapist. 
    You encourage the user to study, manage time effectively, and reduce stress. 
    User Profile:
    Name: {user_profile['name']}
    Major: {user_profile['major']}
    Gender: {user_profile['gender']}
    Main Issues: {user_profile['main_issues']}

    Use supportive language, be concise, and provide motivational tips.
    """

    # Build the message array for the ChatCompletion
    # If you want to feed the short-term summary, you can add it as a context message.
    messages = [
        {"role": "system", "content": system_prompt},
        # Optionally, you can feed in the short-term summary as context:
        {"role": "system", "content": f"Short-Term Summary: {current_short_term_summary}"},
        {"role": "user", "content": user_input}
    ]

    # Call the ChatGPT API
    response = openai.chat.completions.create(
        model=CHATGPT_MODEL,
        messages=messages,
        temperature=0.7
    )

    # Extract the response text
    assistant_reply = response.choices[0].message.content
    return assistant_reply


# ==============================================================
# DEMO / DRIVER CODE
# ==============================================================

if __name__ == "__main__":
    print("---- CA Demo: ChatGPT as Study Presser Therapist ----")
    print("Type 'exit' to end.\n")

    # Summarize the user profile (long-term memory). 
    # You might store or print this once at the start or whenever changed.
    user_profile_summary = summarize_user_profile(user_profile)
    print("=== LONG-TERM MEMORY (User Profile Summary) ===")
    print(user_profile_summary)

    while True:
        # In practice, this user_input might come from your Speech-to-Text UI code.
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting...")
            break

        # Store the user message
        store_message("user", user_input)

        # Generate the assistant's reply
        assistant_reply = chat_with_study_presser(user_input)

        # Store the assistant's reply
        store_message("assistant", assistant_reply)

        # Print the assistant's response
        print(f"Assistant: {assistant_reply}")

        # Update short-term summary
        current_short_term_summary = summarize_short_term_history(full_conversation_log, max_messages=5)
        print("\n[Short-Term Summary updated]\n")

    # When the session ends, you can see the full conversation log for developer analysis:
    # For demonstration, let's just print or save to a JSON file.
    conversation_log_json = json.dumps(full_conversation_log, indent=2)
    # Save to a file if desired:
    with open("memory/conversation_log.json", "w", encoding="utf-8") as f:
        f.write(conversation_log_json)
        
    # Also save the short-term summary and long-term user profile summary
    with open("memory/short_term_summary.txt", "w", encoding="utf-8") as f:
        f.write(current_short_term_summary)
    
    with open("memory/user_profile_summary.txt", "w", encoding="utf-8") as f:
        f.write(user_profile_summary)

    print("\n=== Full Conversation Log saved to conversation_log.json ===")
    print("Goodbye!")
