import streamlit as st
import torch
import time
import speech_recognition as sr
from rag import EnhancedRAG

def speech_to_text():
    """Captures speech and converts it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.toast("🎙 Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        st.toast("🔎 Recognizing...")
        text = r.recognize_google(audio, language="en-in")
        st.toast(f"✅ Recognized: {text}")
        return text
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio")
        return ""
    except sr.RequestError:
        st.error("❌ Could not connect to recognition service")
        return ""

def show_chat_page(mongo_db, user_id, rag_system=EnhancedRAG):
    """Show the main chat interface with speech-to-text integration."""
    st.title("💬 Advanced Chat with Your Documents")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_info = torch.cuda.get_device_properties(0)
            st.success(f"GPU detected: {gpu_info.name} ({gpu_info.total_memory / 1024**3:.1f} GB)")
        else:
            st.warning("No GPU detected. Running in CPU mode.")
        
        llm_model = st.selectbox(
            "LLM Model",
            options=["llama3.2:latest", "llama3:latest", "phi3.5:3.8b", "dolphin-phi:latest"],
            index=0,
            key="chat_llm_model"
        )
        st.session_state.llm_model = llm_model

    st.subheader("Ask Questions About Your Documents")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input("Type your question or use the mic:", key="user_input")

    with col2:
        if st.button("🎙"):
            recognized_text = speech_to_text()
            if recognized_text:
                st.session_state["user_input"] = recognized_text

    if "user_input" in st.session_state and st.session_state["user_input"]:
        st.text_input("Type your question or use the mic:", st.session_state["user_input"], key="input_box")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        if "rag" not in st.session_state:
            st.session_state.rag = rag_system(llm_model_name=llm_model)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = st.session_state.rag.ask(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)

if __name__ == "__main__":
    show_chat_page(None, "test_user")
