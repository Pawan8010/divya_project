import streamlit as st
from runner import run_model

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Copilot", page_icon="🤖", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {background: #0E1117; color: white;}
.block-container {padding-top: 2rem;}

.chat-container {
    max-width: 800px;
    margin: auto;
}

.user {
    background: #1f2937;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}

.bot {
    background: #111827;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🤖 AI Copilot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Next-gen LLM Assistant</p>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.5)
    model = st.selectbox("Model", ["Default", "Advanced"])
    st.markdown("---")
    st.success("System Ready ✅")

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CHAT DISPLAY ----------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask anything...")

# ---------------- PROCESS ----------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Generating response... ⚡"):
        response = run_model(user_input)   # 🔥 CONNECTED HERE

    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()