import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

import os

# -----------------------------------------------------------------------------
# App Configuration & Loading
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Text Detector", page_icon="🕵️‍♂️", layout="centered")

st.markdown("""
<style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Dark Mode Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
    }

    /* Top padding removal */
    .block-container {
        padding-top: 2rem !important;
    }

    /* Glassmorphism for text area */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px;
        color: #f8fafc !important;
        backdrop-filter: blur(10px);
        font-size: 1.05rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.4) !important;
    }

    /* Primary Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white !important;
        border: none;
        border-radius: 30px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5);
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 100%);
    }

    /* Custom Warning Card */
    div[data-testid="stAlert"] {
        background: rgba(234, 179, 8, 0.1);
        border: 1px solid rgba(234, 179, 8, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(5px);
        color: #fef08a;
    }

    /* Title Styling */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        color: #a78bfa !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """
    Load the fine-tuned DistilBERT model and tokenizer.
    Uses @st.cache_resource to load only once per session.
    """
    try:
        # Load the Round 2 model saved as 'detector'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'detector')
        
        # Reconstruct split model if needed
        model_file = os.path.join(model_path, 'model.safetensors')
        if not os.path.exists(model_file):
            parts = [f for f in os.listdir(model_path) if f.startswith('model.safetensors.part')]
            if parts:
                parts.sort(key=lambda x: int(x.split('part')[-1]))
                with open(model_file, 'wb') as outfile:
                    for part in parts:
                        with open(os.path.join(model_path, part), 'rb') as infile:
                            outfile.write(infile.read())
                            
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # We used the base distilbert tokenizer during training
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure the model is saved in a folder named 'detector' in the same directory as this app.")
        return None, None

model, tokenizer = load_model()

# -----------------------------------------------------------------------------
# UI Header
# -----------------------------------------------------------------------------
st.title("🕵️‍♂️ AI Text Detector")
st.subheader("BrightPress Internal Tool")

st.markdown("""
This tool uses a fine-tuned DistilBERT model to classify text as either **Human-Written** or **AI-Generated**. 
Paste an article, opinion piece, or student essay below to analyze it.
""")

# -----------------------------------------------------------------------------
# Mandatory Policy Disclaimer (Deliverable 3 Requirement)
# -----------------------------------------------------------------------------
st.warning("""
**⚠️ MANDATORY USAGE DISCLAIMER**

This tool has a measured false-positive rate. It is mathematically impossible for any statistical AI detector to be 100% accurate, especially against paraphrased or lightly-edited text (adversarial evasion). 

**Policy Rules:**
1. **Never** use this tool as the sole evidence against a student or writer for academic misconduct or contract termination.
2. A "Human" verdict does not guarantee the text wasn't AI-assisted.
3. An "AI" verdict should trigger a **human review** or a conversation, not an automatic penalty.
4. Non-native English writing may trigger higher AI confidence scores due to stylistic regularities.
""")

# -----------------------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------------------
user_input = st.text_area("Paste text here for analysis (minimum 100 characters recommended):", height=250)

if st.button("Analyze Text", type="primary"):
    if not model or not tokenizer:
        st.error("Model not loaded. Please check the 'detector' directory.")
    elif len(user_input.strip()) < 20:
        st.error("Please paste a longer text block for meaningful analysis.")
    else:
        with st.spinner("Analyzing text patterns..."):
            # Tokenize input matching training parameters
            inputs = tokenizer(
                user_input, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=200
            )
            
            # Inference
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=1).squeeze()
            
            # Assuming label 0 = HUMAN, label 1 = AI
            prob_human = probs[0].item()
            prob_ai = probs[1].item()
            
            # Determine verdict based on confidence
            confidence = max(prob_human, prob_ai)
            is_ai = prob_ai > prob_human
            
            st.divider()
            
            # Display Verdict
            if is_ai:
                st.error("### 🤖 Verdict: AI-GENERATED")
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                st.markdown("The model detected strong statistical signatures commonly found in language models.")
            else:
                st.success("### ✍️ Verdict: HUMAN-WRITTEN")
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                st.markdown("The model detected sufficient perplexity and burstiness typical of human writing.")
            
            # Show raw distribution
            st.write("---")
            st.write("#### Raw Probability Distribution")
            col1, col2 = st.columns(2)
            col1.metric("Human Probability", f"{prob_human:.1%}")
            col2.metric("AI Probability", f"{prob_ai:.1%}")


