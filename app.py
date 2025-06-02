import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if missing
for res in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{res}')
    except LookupError:
        nltk.download(res)

# Define model names
model_names = ["Logistic Regression", "Naive Bayes"]

# Load models
model_dict = {
    "Logistic Regression": joblib.load("mental_health_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl")
}


# Load TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text.lower()).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Label mapping
label_map = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality disorder",
    5: "Stress",
    6: "Suicidal"
}

# Streamlit UI
st.title("🧠 Mental Health Detection from Social Media")
st.write("Select a model and enter text to analyze mental health status.")

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
        # Final output
        st.success(f"Predicted Mental Health Status: {label_map[int(prediction)]}")

        # Show prediction probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            df_proba = pd.DataFrame({
                "Condition": [label_map[i] for i in range(len(proba))],
                "Probability": proba
            }).sort_values(by="Probability", ascending=False)
            st.subheader("Prediction Probabilities")
            st.bar_chart(df_proba.set_index("Condition"))

# Optional: Label distribution pie chart
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