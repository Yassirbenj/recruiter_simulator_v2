import speech_recognition as sr
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
import os

import pandas as pd


def stxt(key):
    audio_bytes = audio_recorder(energy_threshold=0.01, pause_threshold=2)
    #audio_bytes = audio_recorder(energy_threshold=(-1.0,1.0), pause_threshold=10)

    if audio_bytes:
        try:
            with st.spinner ("Thinking..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                    temp_audio_file.write(audio_bytes)
                    temp_audio_filename = temp_audio_file.name
                # use the audio file as the audio source
                r = sr.Recognizer()
                with sr.AudioFile(temp_audio_filename) as source:
                    audio = r.record(source)  # read the entire audio file
                # recognize speech using Whisper API
                    try:
                        response=r.recognize_whisper_api(audio, api_key=key)
                        #st.write(f"Transcript: {response}")
                    except sr.RequestError as e:
                        response="Could not request results from Whisper API"
                        st.markdown("=red[response]")
            return response
        except:
            st.markdown(" :red[An error have occured. The recording may be to short. We are not able to transcript correctly your voice. Please try again]",)

def stxt2(key,audio_bytes):
    try:
        with st.spinner ("Thinking..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(audio_bytes)
                temp_audio_filename = temp_audio_file.name
            # use the audio file as the audio source
            r = sr.Recognizer()
            with sr.AudioFile(temp_audio_filename) as source:
                audio = r.record(source)  # read the entire audio file
            # recognize speech using Whisper API
                try:
                    response=r.recognize_whisper_api(audio, api_key=key)
                    st.session_state.prompt=response
                    st.write(f"Transcript: {st.session_state.prompt}")
                except sr.RequestError as e:
                    response="Could not request results from Whisper API"
                    st.markdown("=red[response]")
        return response
    except:
        st.markdown(" :red[An error have occured. The recording may be to short. We are not able to transcript correctly your voice. Please try again]")
