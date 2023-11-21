#pip install openai streamlit streamlit-chat
import openai
import streamlit as st
from streamlit_chat import message
st.set_page_config(page_title="Tec-GPT",page_icon=":books:",layout="wide")
 
openai.api_key = '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'
 
def generate_response(user_input):
    completions = openai.Completion.create(
        engine = 'text-davinci-003',
        prompt = user_input,
        max_tokens = 1024,
        n = 1,
        stop = None,
        temperature = 0.5,
    )
    message = completions.choices[0].text
    return message
 
st.title(' :gear:  :red[Chatbot with GPT] :-1:')
 
if'past_response' not in st.session_state:
    st.session_state.past_response = []
if'past_input' not in st.session_state:
    st.session_state['past_input'] = []
 
def get_input():
    input_text = st.text_input("You:", "Hello,how the world has been treating you?", key='input')
    return input_text
 
user_input = get_input()
if user_input:
    out = generate_response(user_input)
    st.session_state.past_input.append(user_input)
    st.session_state.past_response.append(out)
 
if st.session_state.past_response:
    for i in range(len(st.session_state.past_response)):
       
        message(st.session_state.past_input[i], is_user=True, key=str(i) + '_user')
        message(st.session_state.past_response[i],  key=str(i))