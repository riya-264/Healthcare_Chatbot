import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Download required NLTK data (only once)
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model("chatbot_model.h5")

# Clean and tokenize input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convert sentence to bag of words vector
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict intent using the model
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

# Get response from intents JSON
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Hmm... I didn’t quite get that. Could you rephrase? 🧐"

# Run chatbot
print("✅ Great! Healthcare Chatbot is now running. Type 'quit' to exit.\n")

while True:
    message = input("You: ")
    if message.lower() in ["quit", "exit", "bye", "goodbye", "see you", "good night"]:
        print("Bot: Bye! Take care of your health 🩺💖")
        break

    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)

    if ints:
        confidence = round(ints[0]['probability'] * 100, 2)
        print(f"🔍 Top intent: {ints[0]['intent']} ({confidence}% confidence)")

