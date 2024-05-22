import os
from pathlib import Path
from openai import OpenAI

from credentials.credentials import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

def get_transcription(audio: Path) -> str:
    return client.audio.transcriptions.create(
        model="whisper-1",
        file=audio,
    ).text