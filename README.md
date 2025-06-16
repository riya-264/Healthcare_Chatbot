# 🩺 Healthcare Chatbot

An AI-based chatbot that responds to health-related queries using Natural Language Processing. Built with Python and Streamlit, this project provides a simple interface to explore how chatbots can understand user intents and deliver informative responses.

---

## 🚀 Features

- 💬 Understands and responds to common health-related queries
- 🧠 Trained on intent-based dataset using deep learning
- 🌐 Simple and interactive UI using Streamlit
- 📚 Easy to modify and extend with new intents or responses

---

## 🛠️ Tech Stack

- **Python**
- **NLTK** – for tokenization and text preprocessing  
- **TensorFlow / Keras** – for intent classification  
- **Streamlit** – for web app interface  
- **JSON** – for managing intents dataset

---

## 📁 Project Structure

Healthcare_chatbot/
├── app.py # Main Streamlit app
├── intents.json # Intents dataset (questions and responses)
├── model.h5 # Trained neural network model
├── tokenizer.pickle # Tokenizer used for preprocessing
├── label_encoder.pickle # Encodes intent labels
├── requirements.txt # Python dependencies
├── README.md # You're reading it!


---

## 💻 How to Run Locally

1. **Clone the repo:**
```bash
git clone https://github.com/riya-264/Healthcare_chatbot.git
cd Healthcare_chatbot
