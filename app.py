import streamlit as st
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Setup
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load model and vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Labels
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Streamlit App Config
st.set_page_config(page_title="CleanSpeak", layout="centered")
st.title("  CleanSpeak  ")
st.markdown("CleanSpeak is a smart and lightweight tool that identifies toxic language in user comments using NLP and a trained multi-label classification model.")

# Full-width text input box
with st.container():
    st.markdown("<style>textarea { width: 100% !important; }</style>", unsafe_allow_html=True)
    text_input = st.text_area(label="", placeholder="Enter your comment here...", height=150)

# Centered Predict Button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Clean or Mean?", key="predict_button"):
        if text_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            clean_input = clean_text(text_input)
            transformed = tfidf.transform([clean_input])
            prediction = model.predict(transformed)

            st.subheader("Prediction Results:")

            is_toxic = prediction[0][0]  # 'toxic' label

            if is_toxic == 1:
                # TOXIC box (wide + readable)
                st.markdown("""
                <div style='
                    background-color:#ffe6e6;
                    padding: 20px 30px;
                    border-left: 6px solid #cc0000;
                    border-radius: 8px;
                    margin-bottom: 25px;
                    max-width: 800px;
                    margin-left: auto;
                    margin-right: auto;
                '>
                    <h4 style='color:#cc0000; margin: 0;'>This comment is <b>TOXIC</b></h4>
                </div>
                """, unsafe_allow_html=True)

                # Toxic categories with spacing
                sub_labels = label_cols[1:]
                sub_preds = prediction[0][1:]

                st.markdown("<div style='display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px;'>", unsafe_allow_html=True)
                for label, result in zip(sub_labels, sub_preds):
                    if result == 1:
                        st.markdown(f"""
                        <div style='
                            background-color:#ffcccc;
                            color:#800000;
                            padding:10px 15px;
                            border-radius:6px;
                            font-weight:bold;
                            display:inline-block;
                            margin-bottom: 10px;
                        '>
                            {label.replace("_", " ").title()}: Detected
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                # NOT TOXIC box
                st.markdown("""
                <div style='
                    background-color:#e6f9e6;
                    padding: 20px 30px;
                    border-left: 6px solid #009933;
                    border-radius: 8px;
                    max-width: 800px;
                    margin-left: auto;
                    margin-right: auto;
                '>
                    <h4 style='color:#006600; margin: 0;'>This comment is <b>NOT toxic</b>.</h4>
                </div>
                """, unsafe_allow_html=True)

# Footer credit
st.markdown("""
<hr style="margin-top: 50px; margin-bottom: 10px;">
<div style='text-align: center; font-size: 15px; color: gray;'>
    Made by <b>Jitesh Jhamnani</b>
</div>
""", unsafe_allow_html=True)
