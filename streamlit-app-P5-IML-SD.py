# Importation des bibliothèques nécessaires
import re
import dill
import torch
import pickle
import joblib
import html5lib
import numpy as np
import streamlit as st
import tensorflow as tf
from nltk import pos_tag
import tensorflow_hub as hub
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#import nltk
#import keras
#import sklearn
#from sklearn.pipeline import Pipeline
#from keras.layers import Dense, Dropout, BatchNormalization
#from sklearn.preprocessing import FunctionTransformer
#from transformers import BertModel, BertTokenizer
#nltk.download('omw-1.4')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

# Définition du chemin d'accès aux ressources
path = 'ressources/'


# Chargement des fichiers
vectorizer_CV = joblib.load(path + 'countvectorizer.joblib')
vectorizer_TFIDF = joblib.load(path + 'tfidfvectorizer.joblib')
mlb = joblib.load(path + 'multilabelbinarizer.joblib')

with open(path + 'stop_words.pkl', 'rb') as f:
    stop_words = pickle.load(f)

with open(path + 'top_500_tags.pkl', 'rb') as f:
    top_500_tags = pickle.load(f)

with open(path + 'pipelines.pkl', 'rb') as file:
    pipelines = dill.load(file)


# Définition de l'interface utilisateur
st.markdown(
    "<h1 style='margin-top: 0; padding-top: 0;'>Générateur de tags</h1>",
    unsafe_allow_html=True)
subtitle = '<p style="font-size: 30px;">Projet 5 - OpenClassrooms Parcours IML</p>'
st.markdown(subtitle, unsafe_allow_html=True)


# Sélection du modèle à utiliser
st.sidebar.header("Choisir un modèle")
model_choice = st.sidebar.selectbox(
    "",
    ["SGDClassifier", "CountVectorizer", "TFIDFVectorizer",
     "USE + MLP", "USE + CNN"]
)


# Saisie du titre et du texte à utiliser
title = st.text_input("Collez ici votre titre")
post = st.text_area("Collez ici votre texte :", height=175)

# Génération des tags si l'utilisateur a cliqué sur le bouton et a fourni des données
if st.button("Generate Tags") and title and post:
    user_input = title + " " + post

    if model_choice == "SGDClassifier":
        output = pipelines[model_choice].predict(user_input)
        tags = list(mlb.inverse_transform(output)[0])
        pairs = zip(tags, st.columns(len(tags)))
        for i, (text, col) in enumerate(pairs):
            col.button(label=text, key=f"{text}-{i}", on_click=None)

    elif model_choice == "CountVectorizer" or model_choice == "TFIDFVectorizer":
        output  = pipelines[model_choice].transform(user_input)
        tags = [word for word, _ in output[0][:5]]
        pairs = zip(tags, st.columns(len(tags)))
        for i, (text, col) in enumerate(pairs):
            col.button(label=text, key=f"{text}-{i}", on_click=None)

    elif model_choice == "USE + MLP" or model_choice == "USE + CNN":
        if user_input:
            output = pipelines[model_choice].transform(user_input)
            tags = output[0]
            pairs = zip(tags, st.columns(len(tags)))
            for i, (text, col) in enumerate(pairs):
                col.button(label=text, key=f"{text}-{i}", on_click=None)

# streamlit run C:\Users\simon\Downloads\Durand_Simon_3_code_012023.py
