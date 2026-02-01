import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Sentiment Analysis App")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() != "":
        
        # Transform text
        vect_text = vectorizer.transform([user_input])
        
        # Predict
        result = model.predict(vect_text)

        st.success(f"Sentiment: {result[0]}")
    
    else:
        st.warning("Please enter text")
