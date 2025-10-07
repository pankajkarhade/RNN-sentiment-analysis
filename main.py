## step 1: import all library and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items() }

# load the pre-train model with ReLU activation
model = load_model('simple_RNN_IMDB.h5')

# step 2 : Helper functions
# function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words= text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

import streamlit as st
## streamlit app
st.title('IMDB movie Review sentiment analysis')
st.write('Enter the movie review to classify it as positive or Negative')

# user input 
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input = preprocess_text(user_input)

    ## Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment : {sentiment}')
    st.write(f'Prediction score : {prediction[0][0]}')
else:
    st.write('Please enter the movei review')