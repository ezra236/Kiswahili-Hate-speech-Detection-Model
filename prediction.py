import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords
import string
import tkinter as tk
from tkinter import messagebox

# Define Swahili stopwords
swahili_stopwords = set([
    "na", "ni", "ya", "kwa", "wa", "kama", "hivyo", "vile", "kiasi", "nini", "ambayo", "hiyo",
    "sisi", "yeye", "wote", "katika", "hapa", "lakini", "tu", "mimi", "yangu", "yake", "wale",
    "hii", "huyo", "wetu", "wenye", "pia", "bila", "yote", "kuwa", "kwamba"
])

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in swahili_stopwords])  # Remove stopwords
    return text

# Load the dataset
data = pd.read_csv("output.csv")

# Clean the text data
data['cleaned_text'] = data['Text'].apply(clean_text)

# Prepare the data for model training
x = np.array(data["cleaned_text"])
y = np.array(data["Label"])

cv = CountVectorizer()
X = cv.fit_transform(x)  # Fit the Data

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Function to make a prediction and display the result in a message box
def predict_hate_speech():
    sample = text_entry.get()  # Get the input text from the user
    cleaned_sample = clean_text(sample)
    transformed_sample = cv.transform([cleaned_sample]).toarray()
    prediction = clf.predict(transformed_sample)

    if prediction == 0:
        messagebox.showinfo("Prediction Result", "Hate Speech Detected")
    else:
        messagebox.showinfo("Prediction Result", "No Hate Speech Detected")

# Set up the GUI using Tkinter
root = tk.Tk()
root.title("Swahili Hate Speech Detection")

# Create an input label and entry box
tk.Label(root, text="Enter Swahili Text:").pack(pady=10)
text_entry = tk.Entry(root, width=50)
text_entry.pack(pady=10)

# Create a button to trigger prediction
predict_button = tk.Button(root, text="Analyze", command=predict_hate_speech)
predict_button.pack(pady=10)

# Run the Tkinter application
root.mainloop()
