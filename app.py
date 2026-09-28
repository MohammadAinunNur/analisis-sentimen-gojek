import streamlit as st
import pickle
import re

# Load model dan vectorizer
with open('model_sentimen.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

st.title("📊 Aplikasi Analisis Sentimen Ulasan Pengguna")
st.write("Masukkan ulasan untuk mendeteksi apakah sentimennya **Positif** atau **Negatif**.")

user_input = st.text_area("Tulis ulasan di sini:", "")

if st.button("Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("Mohon masukkan teks ulasan terlebih dahulu!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        st.subheader("Hasil Analisis:")
        if str(prediction).lower() in ['positive', 'positif']:
            st.success(f"Sentimen: **POSITIF** 😊")
        else:
            st.error(f"Sentimen: **NEGATIF** 😡")