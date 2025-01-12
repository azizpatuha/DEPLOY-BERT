import streamlit as st
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf 

def user_home():
    # @st.cache(allow_output_mutation=True)
    def get_model():
        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
        model = TFBertForSequenceClassification.from_pretrained("azizpatuha/BERT")
        return tokenizer, model

    tokenizer, model = get_model()

    user_input = st.text_area('Enter Text to Analyze')
    button = st.button("Analyze")

    d = {
        0: 'negatif',
        1: 'positif'
    }

    if user_input and button:
        test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='tf')
        
        output = model(**test_sample)
        # st.write("Logits: ", output.logits)
        
        y_pred = np.argmax(output.logits.numpy(), axis=1)  # .numpy() untuk mengambil nilai array
        st.write("Prediction: ", d[y_pred[0]])

if __name__ == "__main__":
    user_home()  
