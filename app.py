import streamlit as st

st.set_page_config(
    page_title="Nigerian Pidgin Sentiment Analyzer",
    page_icon="🇳🇬",
    layout="centered"
)

st.title("🇳🇬 Nigerian Pidgin Sentiment Analyzer")

st.write(
    "Analyze the sentiment of Nigerian Pidgin text."
)

text = st.text_area(
    "Enter Nigerian Pidgin text:",
    placeholder="Example: This thing sweet well well!"
)

if st.button("Analyze Sentiment"):
    if text.strip():
        st.success("Text received successfully!")
    else:
        st.warning("Please enter some text.")
