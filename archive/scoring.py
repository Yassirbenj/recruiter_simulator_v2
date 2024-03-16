import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)



def scoring(discussion):
    #uploaded_file = st.file_uploader("upload a evaluation grid file")
    #if uploaded_file is not None:
    #df = pd.read_csv(uploaded_file)
    eval_list=['Preparation','Setting call agenda','Discovery','Positive language','Restate pain points','Illustrate Value','Highlight against competitor','stories','Next steps']
    openai_api_key = st.secrets["openai"]
    chat=ChatOpenAI(model_name='gpt-4',temperature=1,openai_api_key=openai_api_key)
    context = f"evaluate a discussion between a sales person and a customer based on following discussion: {discussion}. give a feedback to the sales person on the good points and the major point to be improved based on the following evaluation parameters: {eval_list}. Give clear explanations for each parameter. "
    context += "Give a grade from 0 to 100% for each of those parameters. Calculate a global grade as the average of the grade of each parameter"
    st.session_state.messages=[]
    st.session_state.messages.append(SystemMessage(content=context))
    st.session_state.messages.append(HumanMessage(content=discussion))
    with st.spinner ("Thinking..."):
        response=chat(st.session_state.messages)
        st.write(response.content)
        #return response.content





        #llm=OpenAI(openai_api_key=openai_api_key,)
        #template = """Question: evaluating a discussion between a sales person and a customer based on following discussion {discussion}. give a feedback to the sales person on the good points and the major point to be improved based on the following evaluation parameters {grid}. Give clear explanations for each parameter. """
        #prompt = PromptTemplate(template=template, input_variables=["discussion","grid"])
        #llm_chain = LLMChain(prompt=prompt, llm=llm)
        #input_list = {"discussion": discussion,"grid": eval_list}
        #response=llm_chain.run(input_list)
        #st.title("Evaluation of the discussion")
        #st.write(response)
        #return response


def scoring_eval():
    uploaded_file = st.file_uploader("upload a evaluation grid file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        eval_list=df.iloc[:,0].tolist()
        #st.write(eval_list)
        return eval_list

discussion=st.file_uploader("upload discussion")
if discussion is not None:
    file_contents = discussion.read()
    decoded_discussion = file_contents.decode('utf-8')
    scoring(decoded_discussion)
