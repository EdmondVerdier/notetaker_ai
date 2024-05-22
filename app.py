from pathlib import Path
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from src.utils.get_transcription import get_transcription
from src.utils.get_answer import get_answer
from src.utils.generate_chunks import generate_chunks

import chromadb
from chromadb.db.base import UniqueConstraintError

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

    chunks = generate_chunks(transcription)
    chroma_client = chromadb.Client()

    # Create a collection if it doesn't exist
    try:
        collection = chroma_client.create_collection(name="ai_notetaker")
    except UniqueConstraintError:
        collection = chroma_client.get_collection(name="ai_notetaker")

    collection.add(documents = chunks,
                    ids = [str(i) for i in range(len(chunks))])
    
    st.write("Chunks:")
    for i, chunk in enumerate(chunks):
        st.write(f"{i}: {chunk}")
    
    question = st.text_input(label="Question", placeholder="Enter a question please")
    if question:
        answers = []
        results = collection.query(
            query_texts=[question], # Chroma will embed this for you
            n_results=2 # how many results to return
        )
        message = ""
        for result in results:
            message += f"Context: {results['documents']}"
        message += f"""
                Question: {question}
                If the answer is not in the context above, please answer with empty string "Not in the context".
                Answer:
            """

        answer = get_answer(message)
        st.write(answer)
