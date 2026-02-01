import streamlit as st
import joblib

# Load model
model = joblib.load("sentiment_model.pkl")

st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="centered"
)

st.title("😊 Sentiment Analysis Web App")

st.write("Enter a sentence and check its sentiment!")

user_input = st.text_area("Your text here:")

if st.button("Analyze"):
    if user_input.strip() != "":
        result = model.predict([user_input])
        st.success(f"Sentiment: {result[0]}")
    else:
        st.warning("Please enter text")
