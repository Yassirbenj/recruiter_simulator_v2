import streamlit as st
import time

def time_counter(duration):
    placeholder=st.empty()
    for seconds in range(duration):
            minutes=int((duration-seconds)/60)
            seconds_rest=(duration-seconds)-(minutes*60)
            with placeholder.container():
                if minutes >=0 and seconds_rest>=0:
                    st.write(f"⏳ {minutes}:{seconds_rest} remaining")
                    time.sleep(1)
    placeholder.text("time finished")
    return True

count_down=time_counter(10)
