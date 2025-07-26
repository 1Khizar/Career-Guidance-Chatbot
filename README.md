# 💼 Career Guidance Chatbot

An AI-powered chatbot that helps students and job-seekers get career advice using Natural Language Processing (NLP) and Machine Learning. Built with Python, Streamlit, and scikit-learn.

---

## 🧠 Features

- 🔎 Understands user questions using vectorization and cosine similarity
- 💬 Provides intelligent career advice from a custom dataset
- 🧹 Cleans and preprocesses user input for better accuracy
- 🖼 User-friendly interface built with Streamlit
- ✅ Gives reliable answers or politely warns when unsure

---

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- scikit-learn
- NLTK
- Joblib

---

## 🧾 Dataset

A CSV file (CareerGuidanceDataset.csv) containing pairs of questions and answers.  
Each question is preprocessed and vectorized to compare with user queries.

---

## 🖥 How to Run Locally

1. Clone this repository:

```bash
git clone https://github.com/1Khizar/Career-Guidance-Chatbot.git
cd Career-Guidance-Chatbot
Install the required packages:

pip install -r requirements.txt

Run the app:
streamlit run chatbot_app.py

🧳 Folder Structure

Career-Guidance-Chatbot/
│
├── CareerGuidanceDataset.csv
├── chatbot_app.py
├── chatbot_model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md


📬 Contact
Developed by Khizar Ishtiaq
📧 Email: khizarishtiaq59@gmail.com
🌐 GitHub: https://github.com/1Khizar

