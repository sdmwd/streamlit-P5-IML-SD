# Importation des bibliothèques nécessaires
import re
import dill
# import torch
# import keras
import pickle
import joblib
import html5lib
import numpy as np
import streamlit as st
# import tensorflow as tf
# import tensorflow_hub as hub
from nltk import pos_tag
# import tensorflow_hub as hub
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#import nltk
#import sklearn
#from sklearn.pipeline import Pipeline
# from keras.layers import Dense, Dropout, BatchNormalization
#from sklearn.preprocessing import FunctionTransformer
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


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


# Définir un dictionnaire de fonctions de modèles et de leurs paramètres associés
model_functions = {
    "SGDClassifier": {"function": pipelines["SGDClassifier"].predict, "num_tags": None},
    "CountVectorizer": {"function": pipelines["CountVectorizer"].transform, "num_tags": 5},
    "TFIDFVectorizer": {"function": pipelines["TFIDFVectorizer"].transform, "num_tags": 5},
#    "USE + CNN": {"function": pipelines["USE + CNN"].transform, "num_tags": None},
}


# Définition de l'interface utilisateur
st.markdown(
    "<h1 style='margin-top: 0; padding-top: 0;'>Générateur de tags</h1>",
    unsafe_allow_html=True)
subtitle = '<p style="font-size: 30px;">Projet 5 - OpenClassrooms Parcours IML</p>'
st.markdown(subtitle, unsafe_allow_html=True)


# Sélection du modèle à utiliser
st.sidebar.header("Choisir un modèle")
model_choice = st.sidebar.selectbox(
    "", list(model_functions.keys())
)


# Saisie du titre et du texte à utiliser
title = st.text_input("Collez ici votre titre :")
post = st.text_area("Collez ici votre texte :", height=250)

# Génération des tags si l'utilisateur a cliqué sur le bouton et a fourni des données
if st.button("Generate Tags") and title and post:

    # Concaténer le titre et le message en une seule chaîne
    user_input = title + " " + post
    button_style = "background-color: black; color: white; border-radius: 5px;"

    # Si le modèle choisi est présent dans le dictionnaire de fonctions de modèles
    if model_choice in model_functions:

        # Récupérer la fonction et le nombre de tags associés au modèle choisi
        model_function = model_functions[model_choice]["function"]
        num_tags = model_functions[model_choice]["num_tags"]

        # Appliquer le modèle choisi à la chaîne d'entrée
        output = model_function(user_input)

        # Extraire les tags prédits de la sortie
        if model_choice == "SGDClassifier":
            tags = list(mlb.inverse_transform(output)[0])
        elif num_tags is None:
            tags = output[0]
        else:
            tags = [word for word, _ in output[0][:num_tags]]

        # Impression des tags
        buttons = "  ".join([f'<button style="{button_style}">{text}</button>' for text in tags])
        st.markdown(buttons, unsafe_allow_html=True)

# streamlit run C:\Users\simon\Downloads\Durand_Simon_3_code_012023.py
