from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#import os
#from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from datetime import datetime


#from langchain.output_parsers import PydanticOutputParser
#from pydantic import BaseModel, Field, validator
#from typing import List

#openapi_key=os.environ.get("OPENAI_API_KEY")

st.set_page_config(page_title="Customer simulator ")
st.header("Customer simulator")

def main():
    openai_api_key = st.secrets["openai"]

    chat=ChatOpenAI(model_name='gpt-4',temperature=0.5,openai_api_key=openai_api_key)

    sent_eval=[]

    if "messages" not in st.session_state:
        st.cache_data.clear()
        conn_pers = st.experimental_connection("gsheets", type=GSheetsConnection, ttl=1)
        personaes=conn_pers.read(worksheet="personae")
        personae=st.selectbox('Select your personae',personaes.iloc[:,0], key="personae")
        start=st.button('Start')
        if start:
            customer_persona=personaes.iloc[personae-1,-2]
            st.session_state.messages=[
                SystemMessage(content=customer_persona)
                ]
            st.session_state.industry=personaes.iloc[personae-1,1]
            st.session_state.position=personaes.iloc[personae-1,2]
            st.session_state.company_size=personaes.iloc[personae-1,3]
            st.session_state.cost=0
            st.session_state.evals=[]
        with st.sidebar:
                st.write(personaes.iloc[personae-1,:-2])


    if prompt := st.chat_input("Start your call with an introduction"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.sidebar:
                st.write(f"Type of contact: Cold call")
                st.write(f"Industry: {st.session_state.industry}")
                st.write(f"Position: {st.session_state.position}")
                st.write(f"Company size: {st.session_state.company_size}")
                st.write(f"Total Cost (USD): {st.session_state.cost}")
                evaluation=evaluate_sentence(prompt)
                st.write(evaluation)
                st.session_state.evals.append({prompt:evaluation})
        with st.spinner ("Thinking..."):
            with get_openai_callback() as cb:
                response=chat(st.session_state.messages)
                st.session_state.cost=round(cb.total_cost,5)
        st.session_state.messages.append(AIMessage(content=response.content))

    messages=st.session_state.get('messages',[])
    discussion=""

    #st.write(messages)

    for i,msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content,is_user=True,key=str(i)+'_saleperson')
            discussion+=f"Sale person: {msg.content}. "
        else:
            message(msg.content,is_user=False,key=str(i)+'_customer')
            discussion+=f"Customer: {msg.content}. "

    if len(messages) > 5:
        evaluate_button=st.button("Evaluate")
        if evaluate_button:
            if discussion=="":
                st.write("No discussion to evaluate")
            elif len(messages) <= 5:
                st.write("The discussion is too short to be evaluated")
            else:
                recap_response=recap(discussion)
                evaluation_response=evaluate(discussion)
                st.title("Recommendations")
                evaluations=st.session_state.evals
                st.write(evaluations)
                st.cache_data.clear()
                conn = st.experimental_connection("gsheets", type=GSheetsConnection, ttl=1)
                df=conn.read()
                last_index=df.iloc[-1,0]

                #get current time
                current_datetime = datetime.now()

                data={
                        "Index":[last_index+1],
                        "User":[""],
                        "Date":[current_datetime],
                        #"Personae":[st.session_state.personae],
                        "Discussion":[discussion],
                        "Evaluation":[evaluation_response],
                        "Recap":[recap_response]
                    }

                data_df=pd.DataFrame(data)
                data_df_updated=pd.concat([df,data_df])
                conn.update(worksheet="evals",data=data_df_updated)
                st.write("Evaluation stored with success")


def parser(evaluation):
    st.write(evaluation)
    split_1=evaluation.split(":")
    good=split_1[2]
    split_3=split_1[3].split("Overall,")
    improve=split_3[0]

    return good,improve

def recap(discussion):
    openai_api_key = st.secrets["openai"]
    llm=OpenAI(openai_api_key=openai_api_key)
    template = """Question: summarize the discussion between customer and sales person based on following discussion {question} """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response=llm_chain.run(discussion)
    st.title("Recap of the discussion")
    st.write(response)
    return response

def evaluate_sentence(sentence):
    openai_api_key = st.secrets["openai"]
    llm=OpenAI(openai_api_key=openai_api_key)
    template = """Question: you are a coach for sales persons. this sentence {question} is from a sales person discussing with a customer. do you have a better formulation that will help to improve the sales process?  explain why"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response=llm_chain.run(sentence)
    return response

def evaluate(discussion):
    openai_api_key = st.secrets["openai"]
    llm=OpenAI(openai_api_key=openai_api_key)
    template = """Question: evaluating a discussion between a sales person and a customer based on following discussion {question}. give a feedback to the sales person on the good points and the major point to be improved """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response=llm_chain.run(discussion)
    st.title("Evaluation of the discussion")
    st.write(response)
    return response

#def reset_conversation():
    #st.write(st.session_state.messages)
    #st.write(st.session_state.messages[0])
    #st.session_state.messages = st.session_state.messages[0]
    #main()


main()
