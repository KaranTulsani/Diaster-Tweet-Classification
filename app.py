import os
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already downloaded
nltk.download('stopwords')

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

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # remove punctuation/numbers
    tokens = text.split()
    tokens = [PorterStemmer().stem(word) for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Streamlit UI
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
