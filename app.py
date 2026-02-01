import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Sentiment Analysis App")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    vect = vectorizer.transform([user_input])
    result = model.predict(vect)
    st.success(f"Sentiment: {result[0]}")
