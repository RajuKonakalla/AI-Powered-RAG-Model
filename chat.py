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
    """Show the main chat interface with enhanced features."""
    
    if "theme" not in st.session_state:
        st.session_state.theme = {
            "primary_color": "#4CAF50",
            "background_color": "#F5F5F5",
            "secondary_color": "#2196F3",
            "text_color": "#ffffff",
            "accent_color": "#FF9800"
        }
    
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {st.session_state.theme["background_color"]};
            color: {st.session_state.theme["text_color"]};
        }}
        .stButton button {{
            background-color: {st.session_state.theme["primary_color"]};
            color: white;
        }}
        .stProgress .st-bo {{
            background-color: {st.session_state.theme["secondary_color"]};
        }}
        .stTextInput input {{
            border-color: {st.session_state.theme["primary_color"]};
        }}
        .stCheckbox label {{
            color: {st.session_state.theme["text_color"]};
        }}
        .stExpander {{
            border-color: {st.session_state.theme["secondary_color"]};
        }}
        .css-1d391kg, .css-12ch3ly {{
            background-color: {st.session_state.theme["background_color"]};
        }}
    </style>
    """, unsafe_allow_html=True)
    
    st.title("💬 Advanced Chat with Your Documents")
    st.markdown("Upload files and ask questions with multiple answer modes")
    
    with st.sidebar:
        st.header("⚙ Configuration")
        
        with st.expander("🎨 UI Theme"):
            st.session_state.theme["primary_color"] = st.color_picker("Primary Color", st.session_state.theme["primary_color"])
            st.session_state.theme["secondary_color"] = st.color_picker("Secondary Color", st.session_state.theme["secondary_color"])
            st.session_state.theme["background_color"] = st.color_picker("Background Color", st.session_state.theme["background_color"])
            st.session_state.theme["text_color"] = st.color_picker("Text Color", st.session_state.theme["text_color"])
            st.session_state.theme["accent_color"] = st.color_picker("Accent Color", st.session_state.theme["accent_color"])
            if st.button("Apply Theme"):
                st.rerun()
        
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_info = torch.cuda.get_device_properties(0)
            st.success(f"GPU detected: {gpu_info.name} ({gpu_info.total_memory / 1024**3:.1f} GB)")
        else:
            st.warning("No GPU detected. Running in CPU mode.")
        
        llm_model = st.selectbox(
            "LLM Model",
            options=["llama3.2:latest", "llama3:latest","phi3.5:3.8b","dolphin-phi:latest","samantha-mistral:latest","dolphin-mistral:latest",],
            index=0,
            key="chat_llm_model"
        )
        st.session_state.llm_model = llm_model
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            index=1,
            key="chat_embedding_model"
        )
        st.session_state.embedding_model = embedding_model
        
        use_gpu = st.checkbox("Use GPU Acceleration", value=gpu_available, key="chat_use_gpu")
        st.session_state.use_gpu = use_gpu
        
        with st.expander("Advanced Options"):
            chunk_size = st.slider("Chunk Size", 100, 2000, 1000, key="chat_chunk_size")
            st.session_state.chunk_size = chunk_size
            
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, key="chat_chunk_overlap")
            st.session_state.chunk_overlap = chunk_overlap
        
        if st.button("Initialize System"):
            with st.spinner("Initializing Enhanced RAG system..."):
                st.session_state.rag = rag_system(
                    llm_model_name=llm_model,
                    embedding_model_name=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    use_gpu=use_gpu and gpu_available
                )
                st.success(f"System initialized with {embedding_model} on {st.session_state.rag.device}")
                time.sleep(1)
                st.rerun()
        
        st.header("📄 Upload Documents")
        
        success, notebooks = mongo_db.get_notebooks(user_id) if mongo_db else (False, [])
        if success and notebooks:
            notebook_options = [("None", None)] + [(nb["name"], nb["_id"]) for nb in notebooks]
            selected_notebook = st.selectbox(
                "Add to Notebook",
                options=notebook_options,
                format_func=lambda x: x[0],
                key="upload_notebook"
            )
            selected_notebook_id = selected_notebook[1] if selected_notebook else None
            
            use_custom_name = st.checkbox("Use custom name", value=False)
            if use_custom_name:
                custom_name = st.text_input("Custom Document Name", placeholder="Enter custom name")
            else:
                custom_name = None
        else:
            st.write("No notebooks available. Create one in the Notebooks section.")
            selected_notebook_id = None
            custom_name = None
            
        uploaded_files = st.file_uploader("Select Files", 
                                         type=["pdf", "docx", "doc", "txt"], 
                                         accept_multiple_files=True,
                                         key="chat_file_uploader")
        
        if uploaded_files and st.button("Process Files"):
            with st.spinner("Processing files..."):
                if selected_notebook_id and mongo_db:
                    for file in uploaded_files:
                        file_type = "unknown"
                        if file.name.lower().endswith('.pdf'):
                            file_type = "pdf"
                        elif file.name.lower().endswith(('.docx', '.doc')):
                            file_type = "docx"
                        elif file.name.lower().endswith('.txt'):
                            file_type = "txt"
                        
                        file.seek(0)
                        mongo_db.save_document_file(
                            file.getbuffer(),
                            file.name,
                            file_type,
                            user_id,
                            selected_notebook_id,
                            custom_name
                        )
                
                if not st.session_state.get('rag'):
                    st.session_state.rag = rag_system(
                        llm_model_name=llm_model,
                        embedding_model_name=embedding_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_gpu=use_gpu and gpu_available
                    )
                
                success = st.session_state.rag.process_files(
                    uploaded_files, 
                    user_id=user_id,
                    mongodb=mongo_db,
                    notebook_id=selected_notebook_id
                )
                
                if success:
                    metrics = st.session_state.rag.get_performance_metrics()
                    if metrics:
                        st.success("Files processed successfully!")
                        with st.expander("💹 Performance Metrics"):
                            st.markdown(f"*Documents processed:* {metrics['documents_processed']} chunks")
                            st.markdown(f"*Index building time:* {metrics['index_building_time']:.2f} seconds")
                            st.markdown(f"*Total processing time:* {metrics['total_processing_time']:.2f} seconds")
                            st.markdown(f"*Memory used:* {metrics['memory_used_gb']:.2f} GB")
                            st.markdown(f"*Device used:* {metrics['device']}")
                        time.sleep(1)
                        st.rerun()
    
    st.subheader("Select Answer Mode")
    
    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = "direct_retrieval"
    
    mode_description = {
        "direct_retrieval": "Directly retrieve answers from documents (fastest)",
        "enhanced_rag": "Enhanced RAG with multi-stage pipeline for improved answers",
        "hybrid": "Hybrid approach combining document retrieval and web search (most comprehensive)"
    }
    
    mode_cols = st.columns(3)
    with mode_cols[0]:
        direct_mode = st.button("📄 Direct Retrieval", 
                               use_container_width=True,
                               help="Fastest mode, directly uses document content to answer")
        st.caption(mode_description["direct_retrieval"])
        
    with mode_cols[1]:
        enhanced_mode = st.button("🔄 Enhanced RAG", 
                                 use_container_width=True,
                                 help="Improves answers with a multi-stage refinement process")
        st.caption(mode_description["enhanced_rag"])
        
    with mode_cols[2]:
        hybrid_mode = st.button("🌐 Hybrid Search", 
                               use_container_width=True,
                               help="Combines document content with simulated web searches")
        st.caption(mode_description["hybrid"])
    
    if direct_mode:
        st.session_state.rag_mode = "direct_retrieval"
    elif enhanced_mode:
        st.session_state.rag_mode = "enhanced_rag"
    elif hybrid_mode:
        st.session_state.rag_mode = "hybrid"
    
    st.info(f"Current mode: {st.session_state.rag_mode} - {mode_description[st.session_state.rag_mode]}")
    
    st.subheader("Ask Questions About Your Documents")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processed_messages" not in st.session_state:
        st.session_state.processed_messages = set()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if isinstance(message["content"], dict):
                    st.markdown(message["content"]["answer"])
                    
                    if "mode" in message["content"]:
                        mode_name = message["content"]["mode"]
                        mode_icons = {
                            "direct_retrieval": "📄",
                            "enhanced_rag": "🔄",
                            "hybrid": "🌐"
                        }
                        icon = mode_icons.get(mode_name, "ℹ")
                        st.caption(f"{icon} Answer mode: {mode_name}")
                    
                    if "query_time" in message["content"]:
                        st.caption(f"⏱ Response time: {message['content']['query_time']:.2f} seconds")
                    
                    if message["content"].get("mode") == "enhanced_rag" and "initial_answer" in message["content"]:
                        with st.expander("🔄 View Enhancement Process"):
                            st.subheader("Initial Answer")
                            st.markdown(message["content"]["initial_answer"])
                            st.divider()
                            st.subheader("Enhanced Answer")
                            st.markdown(message["content"]["answer"])
                    
                    if message["content"].get("mode") == "hybrid":
                        if "doc_sources_count" in message["content"] and "web_sources_count" in message["content"]:
                            st.caption(f"Combined {message['content']['doc_sources_count']} document sources and {message['content']['web_sources_count']} web sources")
                    
                    if "sources" in message["content"] and message["content"]["sources"]:
                        with st.expander("📄 View Sources"):
                            for i, source in enumerate(message["content"]["sources"]):
                                if source.get("file_type") == "web":
                                    st.markdown(f"*Source {i+1}: 🌐 {source['source']}*")
                                else:
                                    st.markdown(f"*Source {i+1}: 📄 {source['source']}*")
                                st.text(source["content"])
                                st.divider()
                else:
                    st.markdown(message["content"])
    
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    
    if "speech_result" not in st.session_state:
        st.session_state.speech_result = ""
    
    if "current_input" not in st.session_state:
        st.session_state.current_input = ""
        
    def update_input():
        st.session_state.current_input = st.session_state[f"user_input_{st.session_state.input_key}"]
    
    def submit_query():
        if st.session_state.current_input:
            user_input = st.session_state.current_input
            
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            st.session_state.current_input = ""
            st.session_state.input_key += 1
            
            st.rerun()
    
    st.markdown("""
    <style>
    /* Style for the input row container */
    .input-container {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Make mic button circular */
    div[data-testid="column"]:nth-of-type(2) .stButton button {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        padding: 0px;
        line-height: 40px;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Align the send button height */
    div[data-testid="column"]:nth-of-type(3) .stButton button {
        height: 40px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        input_value = st.session_state.speech_result if st.session_state.speech_result else st.session_state.current_input
        
        st.text_input(
            "Type your question:",
            value=input_value,
            key=f"user_input_{st.session_state.input_key}",
            on_change=update_input,
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("🎙", key="mic_button"):
            recognized_text = speech_to_text()
            if recognized_text:
                st.session_state.current_input = recognized_text
                st.session_state.speech_result = ""
                st.rerun()
    
    with col3:
        if st.button("Send", key="send_button"):
            submit_query()
    
    current_input = st.session_state.get(f"user_input_{st.session_state.input_key}", "")
    if current_input and current_input != st.session_state.current_input:
        st.session_state.current_input = current_input
        submit_query()
    
    if st.session_state.speech_result:
        st.session_state.speech_result = ""
    
    if st.session_state.messages:
        for i, message in enumerate(st.session_state.messages):
            if isinstance(message["content"], dict):
                content_hash = hash(str(message["content"]))
            else:
                content_hash = hash(message["content"])
            
            message_id = f"{i}_{message['role']}_{content_hash}"
            
            if (message["role"] == "user" and 
                message_id not in st.session_state.processed_messages and
                (i == len(st.session_state.messages) - 1 or 
                 st.session_state.messages[i+1]["role"] != "assistant")):
                
                st.session_state.processed_messages.add(message_id)
                user_input = message["content"]
                
                if "rag" not in st.session_state:
                    st.session_state.rag = rag_system(llm_model_name=st.session_state.get("llm_model", "llama3.2:latest"))
                
                with st.chat_message("assistant"):
                    try:
                        with st.spinner(f"Processing with {st.session_state.rag_mode} mode..."):
                            response = st.session_state.rag.ask(
                                user_input,
                                mode=st.session_state.rag_mode,
                                user_id=user_id,
                                mongodb=mongo_db
                            )
                        
                        if isinstance(response, str) and "Please upload and process documents first" in response:
                            potential_domains = st.session_state.rag.detect_query_domain(user_input)
                            if potential_domains:
                                st.info(f"You seem to be asking about {potential_domains[0]}. Please upload relevant documents first.")
                            else:
                                st.markdown(response)
                            
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            if isinstance(response, dict):
                                st.markdown(response["answer"])
                                
                                if "mode" in response:
                                    mode_name = response["mode"]
                                    mode_icons = {
                                        "direct_retrieval": "📄",
                                        "enhanced_rag": "🔄",
                                        "hybrid": "🌐"
                                    }
                                    icon = mode_icons.get(mode_name, "ℹ")
                                    st.caption(f"{icon} Answer mode: {mode_name}")
                                
                                if "query_time" in response:
                                    st.caption(f"⏱ Response time: {response['query_time']:.2f} seconds")
                                
                                if response.get("mode") == "enhanced_rag" and "initial_answer" in response:
                                    with st.expander("🔄 View Enhancement Process"):
                                        st.subheader("Initial Answer")
                                        st.markdown(response["initial_answer"])
                                        st.divider()
                                        st.subheader("Enhanced Answer")
                                        st.markdown(response["answer"])
                                
                                if response.get("mode") == "hybrid":
                                    if "doc_sources_count" in response and "web_sources_count" in response:
                                        st.caption(f"Combined {response['doc_sources_count']} document sources and {response['web_sources_count']} web sources")
                                
                                if "sources" in response and response["sources"]:
                                    with st.expander("📄 View Sources"):
                                        for i, source in enumerate(response["sources"]):
                                            if source.get("file_type") == "web":
                                                st.markdown(f"*Source {i+1}: 🌐 {source['source']}*")
                                            else:
                                                st.markdown(f"*Source {i+1}: 📄 {source['source']}*")
                                            st.text(source["content"])
                                            st.divider()
                    except Exception as e:
                        error_message = f"Error generating answer: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                
                st.rerun()

if __name__ == "__main__":
    show_chat_page(None, "test_user")