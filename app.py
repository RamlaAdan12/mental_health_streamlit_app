import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import os

# 💡 1. Ensure all needed NLTK resources are available
nltk_packages = ['punkt', 'stopwords', 'wordnet']

for pkg in nltk_packages:
    try:
        if pkg == 'punkt':
            nltk.data.find('tokenizers/punkt')
        else:
            nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# 🔤 NLP preprocessing tools
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 🎯 2. Define model names
model_names = ["Logistic Regression", "Naive Bayes"]

# 📦 3. Load models and vectorizer
try:
    model_dict = {
        "Logistic Regression": joblib.load("mental_health_model.pkl"),
        "Naive Bayes": joblib.load("naive_bayes_model.pkl")
    }
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# 🔧 4. NLP Setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Clean and tokenize
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text.lower()).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

# 🏷 5. Label mapping
label_map = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality disorder",
    5: "Stress",
    6: "Suicidal"
}

# 🌐 6. Streamlit UI
st.set_page_config(page_title="Mental Health Detector", page_icon="🧠")
st.title("🧠 Mental Health Detection from Social Media")
st.write("Select a model and enter text to analyze mental health status.")

# 🔘 Model selection and input
model_choice = st.selectbox("Choose Model", model_names)
user_input = st.text_area("Enter social media post:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        processed_text = preprocess(user_input)
        features = vectorizer.transform([processed_text])
        model = model_dict[model_choice]
        prediction = model.predict(features)[0]
        st.success(f"Predicted Mental Health Status: *{label_map[int(prediction)]}*")

        # 📊 Prediction Probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            df_proba = pd.DataFrame({
                "Condition": [label_map[i] for i in range(len(proba))],
                "Probability": proba
            }).sort_values(by="Probability", ascending=False)
            st.subheader("Prediction Probabilities")
            st.bar_chart(df_proba.set_index("Condition"))

# 📈 7. Label Distribution Pie Chart (static)
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
ax.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%', startangle=90)
ax.axis("equal")
st.subheader("📊 Dataset Label Distribution")
st.pyplot(fig)