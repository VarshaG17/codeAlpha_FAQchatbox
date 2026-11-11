
import streamlit as st
import nltk
import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from faq_data import faqs

# Initialize NLP tools
stemmer = PorterStemmer()

stop_words = {
    "is", "are", "the", "a", "an", "to", "for",
    "of", "in", "on", "at", "do", "does", "what",
    "how", "can", "i", "you", "my", "me", "and",
    "please", "tell", "about"
}

# Text preprocessing
def preprocess(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)

    cleaned_words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)


# Prepare FAQ questions
questions = [faq["question"] for faq in faqs]

processed_questions = [
    preprocess(question) for question in questions
]

# Create TF-IDF model
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(
    processed_questions
)


# Find the best matching answer
def get_answer(user_question):

    if not user_question.strip():
        return "Please enter a question."

    greetings = ["hi", "hello", "hey", "good morning", "good evening"]

    if user_question.lower().strip() in greetings:
        return "Hello! Welcome to the College FAQ Chatbot. How can I help you?"

    processed_input = preprocess(user_question)

    if not processed_input:
        return "Sorry, I couldn't understand your question. Please try again."

    input_vector = vectorizer.transform([processed_input])

    similarities = cosine_similarity(
        input_vector,
        question_vectors
    )

    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score < 0.20:
        return "Sorry, I don't have an answer for that question. Please ask something related to college FAQs."

    return faqs[best_match_index]["answer"]


# Streamlit page settings
st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Page heading
st.title("🎓 College FAQ Chatbot")

st.write(
    "Welcome! I am your AI FAQ assistant. "
    "Ask me anything about college."
)

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# User input
user_input = st.chat_input("Type your question here...")

if user_input:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Generate chatbot response
    answer = get_answer(user_input)

    # Display chatbot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)


# Sidebar
with st.sidebar:

    st.header("About")

    st.write(
        "This chatbot uses NLP techniques "
        "to find answers to frequently asked questions."
    )

    st.subheader("Technologies Used")

    st.write("- Python")
    st.write("- NLTK")
    st.write("- Scikit-learn")
    st.write("- TF-IDF")
    st.write("- Cosine Similarity")
    st.write("- Streamlit")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
