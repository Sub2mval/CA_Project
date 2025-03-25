import re
from playsound import playsound
from gtts import gTTS
import os

def clean_ai_response(response):
    """ Remove AI message headers like ===== AI MESSAGE ===== """
    return re.sub(r"=+\s* Ai Message \s*=+\s*", "", response).strip()
def text_to_speech(text, output_file="data/output/output_2.mp3", lang="en"):
    cleaned_text = clean_ai_response(text)
    print("cleaned = ", cleaned_text)
    tts = gTTS(text=cleaned_text, lang=lang, slow=False)
    tts.save(output_file)

    # Play the generated speech
    playsound(output_file) # Use playSound

# Example usage
# if __name__ == "__main__":
#     text = "I am sorry you feel that way. Let's find a solution together!"
#     text_to_speech(text)
