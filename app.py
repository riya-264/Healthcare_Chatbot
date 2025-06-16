import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')


from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.05
    results = []

    for i, r in enumerate(res):
        if r > ERROR_THRESHOLD:
            results.append({"intent": classes[i], "probability": float(r)})

    results.sort(key=lambda x: x['probability'], reverse=True)

    if not results or results[0]['probability'] < 0.5:
        return [{"intent": "no_match", "probability": 1.0}]

    return results

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Hmm... I didn’t quite get that. Could you rephrase? 🧐"

st.set_page_config(page_title="Healthcare Chatbot", page_icon="🩺")
st.title("🩺 Healthcare Chatbot")
st.markdown("Ask me anything about your symptoms, reports, or health.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ⬅️ The input box that must be outside conditionals
user_input = st.text_input("Type your message here:")

# Show conversation history above the input
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"🧑‍💬 **You**: {message}")
    else:
        st.markdown(f"🤖 **Bot**: {message}")

# Process input
if user_input:
    ints = predict_class(user_input)
    bot_response = get_response(ints, intents)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_response))

    # Optional: force rerun to clear text_input field
    st.rerun()

