from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

def get_llm_response(query,chat_history):

    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:
    
    Chat history : {chat_history}
    
    User question : {user_question}
    
    """

    prompt = ChatPromptTemplate.from_template(template)


    llm = ChatGroq(model="llama-3.1-70b-versatile")


    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history":chat_history,
        "user_question":query
    })


def main():

    st.set_page_config(page_title='Streaming Chatbot',page_icon='🤖')

    st.header("Streaming Chatbot")


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content='Hello, I am a Bot. How can i help you? ')
        ]

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message('Human'):
                st.write(message.content)


    user_input = st.chat_input('Type your message here...')

    if user_input is not None and user_input != "":
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        with st.chat_message("Human"):
            st.markdown(user_input)

        with st.chat_message("AI"):
            response = st.write_stream(get_llm_response(user_input,st.session_state.chat_history))


        st.session_state.chat_history.append(AIMessage(content=response))



if __name__=="__main__":
    main()