import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import re

# 1. Load Dataset sesuai nama file Anda
df = pd.read_csv('GojekAppReviewV4.0.0-V4.9.3_Cleaned.csv')

# Karena kolom teks bernama 'content' dan rating berupa angka di 'score':
df = df.dropna(subset=['content', 'score'])

# 2. Mengubah kolom 'score' menjadi label sentimen (Positif/Negatif)
# Rating 4 & 5 dianggap Positif, Rating 1 & 2 dianggap Negatif (Rating 3 diabaikan/netral)
df = df[df['score'] != 3] # Hilangkan rating 3 jika ingin binari murni
df['sentiment'] = df['score'].apply(lambda x: 'Positive' if x >= 4 else 'Negative')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Menggunakan kolom 'content' untuk teks ulasan
df['cleaned_text'] = df['content'].apply(clean_text)

# 3. Split Data Training dan Testing
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42
)

# 4. Ekstraksi Fitur (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Melatih Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluasi Akurasi
y_pred = model.predict(X_test_vec)
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. Simpan Model dan Vectorizer
with open('model_sentimen.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model berhasil dilatih dan disimpan!")