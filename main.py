import os
import streamlit as st
import sqlite3
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import speech_recognition as sr
import pyttsx3
import threading

# Load API key from environment
load_dotenv()
API_KEY = os.getenv('key')
MODEL_NAME = "compound-beta"

# Initialize ChatGroq model
model = ChatGroq(api_key=API_KEY, model=MODEL_NAME)

# TTS configuration
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1)

# Database configuration
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        message TEXT,
        response TEXT
    )
""")
conn.commit()

def save_chat(user, message, response):
    cursor.execute("INSERT INTO chat_history (user, message, response) VALUES (?, ?, ?)", (user, message, response))
    conn.commit()

def load_chat_history(user):
    cursor.execute("SELECT message, response FROM chat_history WHERE user = ?", (user,))
    return cursor.fetchall()

def text_to_speech(text):
    def speak():
        tts_engine.say(text)
        tts_engine.runAndWait()
    threading.Thread(target=speak).start()

def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        st.info("Transcribing...")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        return "Error with the speech recognition service."

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = FakeEmbeddings(size=1352)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def handle_userinput(user, user_question):
    if not st.session_state.conversation:
        st.error("System error: no conversation chain found.")
        return

    try:
        response = st.session_state.conversation({'question': user_question})
        answer = response.get('answer', '').strip()

        if not answer:
            st.warning("No relevant answer found in the datasheet. Fetching from general LLM...")
            fallback = model.invoke(user_question)
            answer = fallback.content if hasattr(fallback, "content") else str(fallback)

        st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
        text_to_speech(answer)
        save_chat(user, user_question, answer)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    st.set_page_config(page_title="BVEC Query BOT", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("BVEC Query BOT")
    user = st.text_input("Enter your name:")
    user_question = st.text_input("Ask a question:")

    if st.button("Speak"):
        user_question = transcribe_audio()
        st.text(f"**You said:** {user_question}")

    if user and user_question:
        handle_userinput(user, user_question)

    # Automatically process pre-uploaded PDF only once
    if st.session_state.conversation is None:
        datasheet_path = "datasheet.pdf"
        if os.path.exists(datasheet_path):
            with st.spinner("Processing datasheet..."):
                raw_text = get_pdf_text(datasheet_path)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Datasheet loaded and processed!")
        else:
            st.warning("Preloaded 'datasheet.pdf' not found.")

    if user:
        st.subheader("Chat History")
        chat_history = load_chat_history(user)
        for msg, resp in chat_history:
            st.write(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", resp), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
