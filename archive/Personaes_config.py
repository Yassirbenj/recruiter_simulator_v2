import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

st.cache_data.clear()
conn_pers = st.experimental_connection("gsheets", type=GSheetsConnection, ttl=1)
df=conn_pers.read(worksheet="personae")
st.dataframe(df)

update=st.button ("update personae")

if update:
    #prepare the prompt
    df['customer_persona']="You are a customer responding to a call from a sales person. "
    df['customer_persona']+="You are in the industry of "+ df['Customer industry'] + "."
    df['customer_persona']+=" You have the following position in the company: "+df['Customer position']+"."
    df['customer_persona'] += " The size of your company is " + df['Company size'] +"."
    df['customer_persona'] += " Your Existing technical solutions are: " + df['Technical informations'] +"."
    df['customer_persona'] += " The main pain points in your business are "+ df['Customer pain points'] +"."
    df['customer_persona'] += " your decision making factors are " + df['Decision factors'] +"."
    df['customer_persona'] += " your main personality trait are "+ df['Key personality traits'] +"."
    df['customer_persona'] += " you respond briefly to the questions. you do not easily disclose your needs and expectations. you are a customer not an assistant "

    df['email_persona']="You are a customer responding to an email from a sales person. "
    df['email_persona']+="You are in the industry of "+ df['Customer industry'] + "."
    df['email_persona']+=" You have the following position in the company: "+df['Customer position']+"."
    df['email_persona'] += " The size of your company is " + df['Company size'] +"."
    df['email_persona'] += " Your Existing technical solutions are: " + df['Technical informations'] +"."
    df['email_persona'] += " The main pain points in your business are "+ df['Customer pain points'] +"."
    df['email_persona'] += " your decision making factors are " + df['Decision factors'] +"."
    df['email_persona'] += " your main personality trait are "+ df['Key personality traits'] +"."
    #df['email_persona'] += " you respond briefly to the question. you are a customer not an assistant "

    conn_pers.update(worksheet="personae",data=df)
    st.write("Evaluation stored with success")
