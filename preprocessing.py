# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords
import string

# Define a custom list of Swahili stopwords (extend as necessary)
swahili_stopwords = set([
    "na", "ni", "ya", "kwa", "wa", "kama", "hivyo", "vile", "kiasi", "nini", "ambayo", "hiyo",
    "sisi", "yeye", "wote", "katika", "hapa", "lakini", "tu", "mimi", "yangu", "yake", "wale",
    "hii", "huyo", "wetu", "wenye", "pia", "bila", "yote", "kuwa", "kwamba"
])

# Function to clean the text (remove punctuation, stopwords, etc.)
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in swahili_stopwords])  # Remove stopwords
    return text

# Load the CSV file containing the Swahili dataset
data = pd.read_csv("output.csv")

# Clean the 'Text' column (assuming the column with the text is named 'Text')
data['cleaned_text'] = data['Text'].apply(clean_text)

# Print the first few rows of the cleaned data
print(data[['Text', 'cleaned_text']])

