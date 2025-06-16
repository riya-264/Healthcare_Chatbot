import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lists
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Loop through all patterns and tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Lowercase and tokenize pattern
        wordList = nltk.word_tokenize(pattern.lower())
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean words list
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))  # ✅ FIX: you had set(classes) by mistake

classes = sorted(set(classes))  # correct

# Save processed data
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle and convert to numpy arrays
random.shuffle(training)
training = np.array(training)

trainX = np.array(training[:, :len(words)], dtype=np.float32)
trainY = np.array(training[:, len(words):], dtype=np.float32)

# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Compile and train
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Save model and print confirmation
model.save('chatbot_model.h5', hist)
print("✅ Training complete. Model saved as chatbot_model.h5")
