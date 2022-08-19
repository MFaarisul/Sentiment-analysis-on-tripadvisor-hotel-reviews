import string
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import streamlit as st
# Preprocessing and evaluation
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#Model
from sklearn.linear_model import LogisticRegression

@st.cache
def data():
    df = pd.read_csv('Deployment/cleaned_df.csv')
    X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Rating'], test_size=0.2)

    return X_train, X_test, y_train, y_test

@st.cache
def ml_prep():
    X_train, _, y_train, _ = data()

    # ML
    tfid = TfidfVectorizer()
    train_tfid_matrix = tfid.fit_transform(X_train)

    log = LogisticRegression(max_iter=1000)
    log.fit(train_tfid_matrix, y_train)

    return tfid, log

@st.cache(hash_funcs={tf.keras.models.Sequential: id})
def dl_prep():
    _, _, y_train, _ = data()

    # DL 
    tokenizer = pickle.load(open('Deployment/tokenizer.pkl', 'rb'))
    model = tf.keras.models.load_model('Deployment/dl_model.h5')
    
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(y_train)

    return tokenizer, model, lb

def cleaning(text):
    #remove punctuations and uppercase
    clean_text = text.translate(str.maketrans('','',string.punctuation)).lower()
    
    #remove stopwords
    clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]
    
    #lemmatize the word
    sentence = []
    for word in clean_text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))

    return ' '.join(sentence)

# Logistic Regression
def ml_predict(text):
    tfid, log = ml_prep()

    clean_text = cleaning(text)
    tfid_matrix = tfid.transform([clean_text])
    pred_proba = log.predict_proba(tfid_matrix)
    idx = np.argmax(pred_proba)
    pred = log.classes_[idx]
    
    return pred, pred_proba[0][idx]

# Deep Neural Network
def dl_predict(text):
    tokenizer, model, lb = dl_prep()

    clean_text = cleaning(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq)

    pred = model.predict(padded)
    idx = np.argmax(pred)
    # Get the label name back
    result = lb.inverse_transform(pred)[0]
    
    return result, pred[0][idx]

def main():
    html_temp = '''
    <h1 style="font-family: Trebuchet MS; padding: 12px; font-size: 30px; color: #c9184a; text-align: center;
    line-height: 1.25;">Sentiment Analysis<br>
    <span style="color: #ff97b7; font-size: 48px"><b>TripAdvisor Hotel Reviews</b></span><br>
    <span style="color: #ffc8dd; font-size: 20px">Using Sklearn and Tensorflow</span>
    </h1>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)

    models = st.radio('Select Model', ('Logistic Regression', 'Deep Neural Network'))

    text = st.text_area('Write a Review', '')

    if st.button('Analyze'):
        if models == 'Logistic Regression':
            label, probability = ml_predict(text)
        else:
            label, probability = dl_predict(text)

        if label == 'Good':
            st.info('Probability: {:.2f}'.format(probability))
            st.success('The sentiment for this particular review is Good')
        elif label == 'Bad':
            st.info('Probability: {:.2f}'.format(probability))
            st.error('The sentiment for this particular review is Bad')
        else:
            st.info('Probability: {:.2f}'.format(probability))
            st.warning('The sentiment for this particular review is neither Good nor Bad')

if __name__ == '__main__':
    main()