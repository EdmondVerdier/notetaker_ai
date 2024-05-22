from pathlib import Path
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import chromadb

from src.utils.get_transcription import get_transcription
from src.utils.get_answer import get_answer
from src.utils.generate_chunks import generate_chunks
from src.utils.initialise_vector_store import initialise_vector_store

st.header("AI Note Taker 🤖")
st.write("Welcome to the AI Note Taker! 🎉")

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = ''

AUDIO_PATH = Path("audio.wav")
COLLECTION_NAME = "ai_notetaker"

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

    collection = initialise_vector_store(chroma_client, chunks)
    
    # Display the chunks
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
                You are a helpful assistant. You will be given a context and a discussion. Answer only using this information.
                If you can't find the answer, please say "I don't know".
                Answer:
            """
        
        st.session_state["message_history"] += f"\n\n {message}"

        answer = get_answer(st.session_state["message_history"])
        st.session_state["message_history"] += answer

        st.write(answer)
