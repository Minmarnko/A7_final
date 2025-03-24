import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- CONFIGURE PAGE ---
st.set_page_config(
    page_title="Text Guard - Hate Speech Detector",
    page_icon="🧠",
    layout="centered"
)

# --- LOAD MODEL ---
MODEL_PATH = "best_model_odd_student"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

model, tokenizer = load_model()

# --- LABELS ---
LABELS = ["✅ Clean", "⚠️ Offensive", "🚫 Hate Speech"]

# --- CUSTOM STYLING ---
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f0f2f6;
    }
    .main-title {
        font-size: 2.5em;
        font-weight: 800;
        color: #4B0082;
        margin-bottom: 0.2em;
    }
    .subheader {
        font-size: 1.1em;
        color: #555;
        margin-bottom: 1em;
    }
    .result-box {
        background-color: #fff;
        border-left: 6px solid #4B0082;
        padding: 1.5em;
        margin-top: 2em;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .label {
        font-size: 1.4em;
        font-weight: bold;
        color: #4B0082;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- HEADER ---
st.markdown('<div class="main-title">🧠 Text Guard</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">AI-powered detection for hate and offensive language</div>', unsafe_allow_html=True)
st.markdown("---")

# --- TEXT INPUT ---
user_input = st.text_area("🔎 Type a sentence for analysis:", "", height=130, placeholder="e.g., I can't stand people like you!")

# --- PREDICTION ---
if st.button("🛡️ Run Analysis"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Prepare input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        # Display result
        st.markdown(f"""
        <div class="result-box">
            <div class="label">Prediction Result:</div>
            <div style="font-size: 2em; margin-top: 10px;">{LABELS[predicted_class]}</div>
        </div>
        """, unsafe_allow_html=True)
