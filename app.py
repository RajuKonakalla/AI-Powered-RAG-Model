import streamlit as st


st.set_page_config(
        page_title="GPU-Accelerated RAG System",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

import os
import atexit

from database import MongoDB
from auth import show_login_page, show_signup_page, check_session, logout_user
from rag import EnhancedRAG
from notebooks import show_notebooks_page, show_notebook_detail_page, show_document_view_page
from settings import show_settings_page
from chat import show_chat_page
from utils import init_session_state, cleanup_temp_files, set_page_style

st.markdown("""
<style>
    /* General Styles */
    .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)), 
                        url("https://cdn.dribbble.com/userupload/32690165/file/original-91b920c5af3417e456bc46004300944d.gif") no-repeat center center ;
            background-size: cover;
    }

    /* Text Input Styling */
    .stTextInput>div>div>input {
        background-color: transparent; /* Dark input background */
        color: white; /* Neon text color */
        border: 2px solid #00ffcc; /* Neon border */
        border-radius: 5px; /* Rounded corners */
        padding: 10px; /* Padding for better spacing */
    }

    /* Button Styling */
    

    /* Button Hover Effect */
    .stButton>button:hover {
        background-color: transparent; 
        transition: background-color 0.3s ease-in-out;
    }

    /* Chat Bubble Styles */
    .chat-bubble {
        padding: 10px; /* Padding inside the bubble */
        border-radius: 10px; /* Rounded corners */
        margin: 5px 0; /* Spacing between bubbles */
        max-width: 70%; /* Limit bubble width */
    }

    /* User Chat Bubble */
    .user-bubble {
        background-color: #00ffcc; /* Neon green background */
        color: #0d0d0d; /* Dark 
             color */
        align-self: flex-end; /* Align to the right */
    }

    /* Bot Chat Bubble */
    .bot-bubble {
        background-color: #ff00ff; /* Neon magenta background */
        color: #0d0d0d; /* Dark text color */
        align-self: flex-start; /* Align to the left */
    }

    /* Glowing Effect */
    .glow {
        box-shadow: 0 0 10px #00ffcc, 0 0 20px #00ffcc, 0 0 30px #00ffcc; /* Neon glow */
    }

    /* Split-Screen Layout */
    .split-screen {
        display: flex; /* Flexbox for layout */
        gap: 20px; /* Spacing between panels */
    }

    /* Left and Right Panels */
    .left-panel, .right-panel {
        flex: 1; /* Equal width for both panels */
    }

    /* Progress Bar Animation */
    .progress-bar {
        width: 100%; /* Full width */
        background-color: #1a1a1a; /* Dark background */
        border-radius: 10px; /* Rounded corners */
        overflow: hidden; /* Hide overflow */
        margin: 10px 0; /* Spacing */
    }

    .progress-bar-fill {
        height: 20px; /* Height of the progress bar */
        background-color: #00ffcc; /* Neon fill color */
        box-shadow: 0 0 10px #00ffcc, 0 0 20px #00ffcc; /* Glow effect */
        animation: glow 1.5s infinite alternate; /* Glowing animation */
    }

    /* Glow Animation Keyframes */
    @keyframes glow {
    0% { box-shadow: 0 0 5px #00ffcc; }
    100% { box-shadow: 0 0 15px #00ffcc; }
}

    /* Apply glow only on hover */
    .stTextInput>div>div>input:hover,
    .stButton>button:hover {
        animation: glow 1s infinite alternate;
    }

    /* Smooth Transitions */
    .fade-in {
        animation: fadeIn 1s ease-in-out; /* Fade-in animation */
    }

    /* Fade-In Animation Keyframes */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
            
</style>
""", unsafe_allow_html=True)
def main():
    """Main application entry point."""
    
    set_page_style()
    
    init_session_state()
    
    if not st.session_state.mongo_db:
        connection_string = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
        st.session_state.mongo_db = MongoDB(connection_string)
    
    logged_in = check_session(st.session_state.mongo_db)
    
    if not logged_in and st.session_state.user is None:
        st.markdown(
            """<h1 style="color: lightblue; font-family: 'Ceveat', cursive;">
                🚀 Advanced RAG System with Multiple Modes
            </h1>""",
            unsafe_allow_html=True
        )
        
        if st.session_state.auth_page == "login":
            show_login_page(st.session_state.mongo_db)
        else:
            show_signup_page(st.session_state.mongo_db)
        return
    
    with st.sidebar:
        st.title("WELCOME ")
        st.markdown(f"Welcome, **{st.session_state.user['name']}**!")
        
        st.header("📌 Navigation")
        nav_options = {
            "chat": "💬 Chat",
            "notebooks": "📚 Notebooks",
            "settings": "⚙️ Settings"
        }
        
        selected_nav = st.radio(
            "Go to",
            options=list(nav_options.keys()),
            format_func=lambda x: nav_options[x],
            key="nav_selection",
            index=list(nav_options.keys()).index(st.session_state.page) if st.session_state.page in nav_options else 0
        )
        
        if selected_nav != st.session_state.page and st.session_state.page in nav_options:
            st.session_state.page = selected_nav
            st.rerun()
        
        st.button("Logout", on_click=logout_user, args=(st.session_state.mongo_db,))
    
    if st.session_state.page == "notebooks":
        show_notebooks_page(st.session_state.mongo_db, st.session_state.user['user_id'])
    elif st.session_state.page == "notebook_detail":
        show_notebook_detail_page(
            st.session_state.mongo_db, 
            st.session_state.user['user_id'],
            EnhancedRAG
        )
    elif st.session_state.page == "document_view":
        show_document_view_page(st.session_state.mongo_db, st.session_state.user['user_id'])
    elif st.session_state.page == "settings":
        show_settings_page(st.session_state.mongo_db, st.session_state.user['user_id'])
    else:
        show_chat_page(
            st.session_state.mongo_db, 
            st.session_state.user['user_id'],
            EnhancedRAG
        )

if __name__ == "__main__":
    atexit.register(cleanup_temp_files)
    
    main()