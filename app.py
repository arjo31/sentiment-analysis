import streamlit as st
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

st.title('Sentiment Analysis using DistilBert')

input_text = st.text_input("Enter your text for sentiment analysis")
emotions = {
    0:'sadness',
    1:'joy',
    2:'love',
    3:'anger',
    4:'fear',
    5:'surprise'
}

if input_text:

    @st.cache_resource(show_spinner=False)
    def load_model_tokenizer(model_path:str):
        model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        return (model, tokenizer)
    
    model, tokenizer = load_model_tokenizer("model")

    predict_input = tokenizer.encode(input_text, truncation=True, padding=True, return_tensors='tf')
    output = model(predict_input)[0]
    pred_value = tf.argmax(output, axis = -1).numpy()[0]
    st.success(f'Predicted sentiment : {emotions[pred_value]}')