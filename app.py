import os
import streamlit as st
import pickle
import re
from nltk.stem import PorterStemmer  # still useful for stemming

# --- Custom Stopwords (replace nltk.stopwords) ---
custom_stopwords = {
    "a", "an", "the", "and", "or", "in", "on", "at", "for", "with",
    "without", "about", "to", "from", "by", "of", "is", "am", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "but", "if", "because", "as", "until", "while",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "them", "we", "you", "i", "me", "my", "mine", "yours", "ours"
}

# Paths to saved model and vectorizer
MODEL_PATH = "models/disaster_tweet_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Check if files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("❌ Model or vectorizer file not found. Please check the 'models/' folder.")
    st.stop()

# Load saved model and vectorizer
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# --- Preprocessing function ---
def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  
    text = re.sub(r'@\w+|#', '', text)  
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = text.split()
    tokens = [PorterStemmer().stem(word) for word in tokens if word not in custom_stopwords]
    return " ".join(tokens)

# --- Streamlit UI ---
st.title("🌪 Disaster Tweet Classifier")
st.write("Enter a tweet and find out if it’s about a disaster or not.")

tweet = st.text_area("Enter your tweet:")

if st.button("Classify"):
    if tweet.strip():
        processed_tweet = preprocess_text(tweet)
        vectorized_tweet = vectorizer.transform([processed_tweet])
        prediction = model.predict(vectorized_tweet)[0]
        
        if prediction == 1:
            st.success("🚨 This tweet is about a disaster.")
        else:
            st.info("✅ This tweet is NOT about a disaster.")
    else:
        st.warning("Please enter a tweet before classifying.")

st.markdown("---")
st.caption("Model trained using Logistic Regression / Random Forest with TF-IDF features.")
