import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from predict import predict_sentiment

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Nigerian Pidgin Sentiment Analyzer",
    page_icon="🇳🇬",
    layout="centered"
)

st.title("🇳🇬 Nigerian Pidgin Sentiment Analyzer")
st.write("Type any Nigerian Pidgin text and get a sentiment prediction.")

# ============================================================
# EXAMPLES
# ============================================================

examples = [
    "This food sweet die!",
    "Abeg this thing don tire me",
    "The meeting go hold by 3pm",
    "You try well well, God bless you",
    "Wetin dey happen na, this network no good at all"
]

st.write("**Try an example:**")
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(f"Example {i+1}", key=f"ex_{i}"):
        st.session_state["text_input"] = ex

# ============================================================
# INPUT
# ============================================================

text = st.text_area(
    "Enter Nigerian Pidgin text:",
    value=st.session_state.get("text_input", ""),
    height=100
)

analyze = st.button("Analyze Sentiment", type="primary")

# ============================================================
# PREDICTION
# ============================================================

if analyze:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        label, confidence, probs = predict_sentiment(text)

        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
        st.markdown(f"## {emoji.get(label, '')} {label.upper()}")
        st.write(f"**Confidence:** {confidence * 100:.1f}%")

        st.write("**Probability breakdown:**")
        for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
            st.write(f"{cls.capitalize()}")
            st.progress(float(p))
            st.caption(f"{p * 100:.1f}%")

# ============================================================
# MODEL INFO
# ============================================================

with st.expander("About this model"):
    st.markdown("""
    - **Model:** TF-IDF (word 1-2 gram + char 3-5 gram) + Logistic Regression
    - **Test accuracy:** 68.2%
    - **Weighted F1:** 0.65
    - **Known limitation:** the neutral class is underrepresented in training
      data and the model struggles to detect it reliably. Predictions lean
      toward negative/positive when text is ambiguous.
    """)
