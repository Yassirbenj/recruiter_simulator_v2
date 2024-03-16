import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd


# Create a connection object.

st.cache_data.clear()
conn = st.experimental_connection("gsheets", type=GSheetsConnection, ttl=1)

df=conn.read()
st.dataframe(df)
