from pathlib import Path
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from src.utils.get_transcription import get_transcription
from src.utils.get_answer import get_answer
from src.utils.generate_chunks import generate_chunks

import chromadb

st.header("AI Note Taker 🤖")
st.write("Welcome to the AI Note Taker! 🎉")

AUDIO_PATH = Path("audio.wav")

audio_bytes = audio_recorder()
if audio_bytes:
    with open(AUDIO_PATH, mode='wb') as f:
        f.write(audio_bytes)
    st.audio(audio_bytes, format="audio/wav")

    st.write("Transcription:")
    transcription = get_transcription(AUDIO_PATH)
    st.write(transcription)
    
    question = st.text_input(label="Question", placeholder="Enter a question please")
    if question:
        answers = []
        for chunk in generate_chunks(transcription):
            message = f"""
                Context: {chunk},
                Question: {question}
                If the answer is not in the context above, please answer with empty string "".
                Answer:
            """
            answers.append(get_answer(message))
        
        # Join the answers into a single string
        answer = " ".join(answers)
        st.write(answer)
