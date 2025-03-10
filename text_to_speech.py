from gtts import gTTS
import os

def text_to_speech(text, output_file="data/output/output_2.mp3", lang="en"):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_file)
    print(f"Audio saved as {output_file}")

# Example usage
if __name__ == "__main__":
    text = "I am sorry you feel that way. Let's find a solution together!"
    text_to_speech(text)
