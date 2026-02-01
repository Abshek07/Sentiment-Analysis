import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\veeks\Desktop\tc\sentiment_dataset.csv")

# Drop rows with missing values
df = df.dropna()

# Optional: Add more rows directly in code
extra_data = pd.DataFrame([
    ["good product", "Positive"],
    ["this is a good product", "Positive"],
    ["very good quality", "Positive"],
    ["highly recommend this good product", "Positive"]
], columns=["Feedback", "Sentiment"])

df = pd.concat([df, extra_data], ignore_index=True)

# Features and labels
X = df["Feedback"]
y = df["Sentiment"]

# Vectorization with TF-IDF (bigrams included)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vect = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("Training complete. Accuracy:", accuracy_score(y_test, y_pred))

# Get input from user
user_input = input("\nEnter your feedback: ")

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Predict
input_vect = vectorizer.transform([user_input])
prediction = model.predict(input_vect)
print("Sentiment:", prediction[0])

import joblib

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved!")

