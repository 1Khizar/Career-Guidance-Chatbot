import streamlit as st
import pandas as pd 
import joblib
import string
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

# Setup
st.set_page_config(page_title="Career Guidance Chatbot", layout="centered")
st.title('🎓 Career Guidance Chatbot')


# Load the CSV saved from Jupyter
df = pd.read_csv('CareerGuidanceDataset.csv')

english_stopwords = stopwords.words('english')

model = joblib.load('chatbot_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def clean_text(text):
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, '')
    return ' '.join([word for word in text.split() if word not in english_stopwords ])

df['cleaned_question'] = df['question'].apply(clean_text)
dataset_ques = vectorizer.transform(df['cleaned_question'])

with st.form("chat_form"):
    user_ques = st.text_input('Ask me a career-related question: ')
    submitted = st.form_submit_button("🔍 Get Answer")

if submitted and user_ques:
    clean_que = clean_text(user_ques)
    user_question_vec = vectorizer.transform([clean_que])

    # Only need to transform once
    similarity_scores = cosine_similarity(user_question_vec, dataset_ques)

    best_index = similarity_scores.argmax()
    best_score = similarity_scores[0, best_index]

    if best_score > 0.6:
        # print("Answer:", df['answer'][best_index])
        st.success("Chatbot Answer: " + df['answer'][best_index])
    else:
        st.error("\nChatbot: Sorry, I don't have enough information about that.")