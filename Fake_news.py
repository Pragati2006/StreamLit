import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.title('📰 Fake News Detection using Logistic Regression')

df = pd.read_csv('news.csv')
df.dropna(inplace=True)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'].str.strip().str.lower())

# Clean text
import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# User input
news_input = st.text_area("Enter the news text to analyze for fake/real:")

if st.button('Analyze'):
    cleaned_input = clean_text(news_input)
    input_tfidf = tfidf.transform([cleaned_input])
    pred = model.predict(input_tfidf)
    if pred[0] == 1:
        st.error('🚨 The news is predicted as FAKE.')
    else:
        st.success('✅ The news is predicted as REAL.')
