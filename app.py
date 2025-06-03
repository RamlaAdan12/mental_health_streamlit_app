import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 💡 1. Ensure all needed NLTK resources are available
nltk_packages = ['punkt_tab', 'stopwords', 'wordnet']
for pkg in nltk_packages:
    try:
        if pkg == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab/english/')
        else:
            nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# 🔤 NLP preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Clean text (match notebook preprocessing)
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)  # Remove URLs, mentions, hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# 🎯 2. Define model names
model_names = ["Logistic Regression", "Random Forest", "Naive Bayes"]

# 📦 3. Load models and vectorizer
try:
    model_dict = {
        "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "Naive Bayes": joblib.load("naive_bayes_model.pkl")
    }
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# 🏷 4. Label mapping
label_map = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality disorder",
    5: "Stress",
    6: "Suicidal"
}

# 🌐 5. Streamlit UI
st.set_page_config(page_title="Mental Health Detector", page_icon="🧠", layout="wide")
st.title("🧠 Mental Health Detection from Social Media")
st.write("Select a model and enter text to analyze mental health status.")
st.markdown("*Note*: This tool is for research purposes only. For professional mental health advice, consult a licensed professional.")

# 🔘 Model selection and input
model_choice = st.selectbox("Choose Model", model_names)
user_input = st.text_area("Enter social media post:", placeholder="e.g., I feel so overwhelmed and can't sleep at night.")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Preprocess and vectorize input
        processed_text = preprocess(user_input)
        if not processed_text:
            st.warning("No valid words found after preprocessing. Try different text.")
        else:
            features = vectorizer.transform([processed_text]).toarray()
            model = model_dict[model_choice]
            prediction = int(model.predict(features)[0])  # Ensure integer prediction
            st.success(f"Predicted Mental Health Status: *{label_map[prediction]}*")

            # 📊 Prediction Probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                df_proba = pd.DataFrame({
                    "Condition": [label_map[i] for i in range(len(proba))],
                    "Probability": proba
                }).sort_values(by="Probability", ascending=False)
                st.subheader("Prediction Probabilities")
                st.bar_chart(df_proba.set_index("Condition"))
            else:
                st.info("Selected model does not support probability predictions.")

# 📈 6. Label Distribution Pie Chart (static)
label_counts = {
    "Normal": 16343,
    "Depression": 15404,
    "Suicidal": 10652,
    "Anxiety": 3841,
    "Bipolar": 2777,
    "Stress": 2587,
    "Personality disorder": 1077
}

fig, ax = plt.subplots()
ax.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494'])
ax.axis("equal")
st.subheader("📊 Dataset Label Distribution")
st.pyplot(fig)