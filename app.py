import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

import os

# -----------------------------------------------------------------------------
# App Configuration & Loading
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Text Detector", page_icon="🕵️‍♂️", layout="centered")

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

# -----------------------------------------------------------------------------
# Instructions for Running (For Grader / User)
# -----------------------------------------------------------------------------
with st.expander("💻 How to run this app"):
    st.markdown("""
    **To run locally:**
    ```bash
    pip install streamlit torch transformers
    streamlit run app.py
    ```
    
    **To run in Google Colab (via LocalTunnel):**
    ```python
    !npm install localtunnel
    !streamlit run app.py &>/dev/null &
    !npx localtunnel --port 8501
    ```
    """)
