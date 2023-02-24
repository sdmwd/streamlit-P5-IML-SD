# Importation des bibliothèques nécessaires
import re
import dill
import nltk
import joblib
import html5lib
import numpy as np
import streamlit as st
from nltk import pos_tag
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
# import torch
# import keras
# import sklearn
# import tensorflow as tf
# import tensorflow_hub as hub
# from keras.layers import Dense, Dropout, BatchNormalization


# Définition du chemin d'accès aux ressources
path = 'ressources/'


# Chargement des fichiers
vectorizer_CV = joblib.load(path + 'countvectorizer.joblib')
vectorizer_TFIDF = joblib.load(path + 'tfidfvectorizer.joblib')
mlb = joblib.load(path + 'multilabelbinarizer.joblib')

with open(path + 'stop_words.pkl', 'rb') as f:
    stop_words = dill.load(f)

with open(path + 'top_500_tags.pkl', 'rb') as f:
    top_500_tags = dill.load(f)

with open(path + 'pipelines.pkl', 'rb') as file:
    pipelines = dill.load(file)


# Définir un dictionnaire de fonctions de modèles et de leurs paramètres associés
model_functions_supervised = {
    "": None,
    "SGDClassifier": {"function": pipelines["SGDClassifier"].predict},
}

model_functions_unsupervised = {
    "": None,
    "CountVectorizer": {"function": pipelines["CountVectorizer"].transform},
    "TFIDFVectorizer": {"function": pipelines["TFIDFVectorizer"].transform}
}

# Définition de l'interface utilisateur
st.markdown(
    "<h1 style='margin-top: 0; padding-top: 0;'>Générateur de tags</h1>",
    unsafe_allow_html=True)
subtitle = '<p style="font-size: 30px;">Projet 5 - OpenClassrooms Parcours IML</p>'
st.markdown(subtitle, unsafe_allow_html=True)


# Sélection du modèle à utiliser
st.sidebar.header("Choisir un modèle")

def on_select():
    with st.sidebar.container():
        unsupervised_choice = st.selectbox("Approche non supervisée", model_functions_unsupervised.keys(), key=3)

with st.sidebar.container():
    supervised_choice = st.selectbox("Approche supervisée", model_functions_supervised.keys(), on_change=on_select, key=1)

with st.sidebar.container():
    unsupervised_choice = st.selectbox("Approche non supervisée", model_functions_unsupervised.keys(), key=2)



# Saisie du titre et du texte à utiliser
title = st.text_input("Collez ici votre titre :")
post = st.text_area("Collez ici votre texte :", height=250)


# # Génération des tags si l'utilisateur a cliqué sur le bouton et a fourni des données
# if st.button("Generate Tags") and title and post:
#
#     # Concaténer le titre et le message en une seule chaîne
#     user_input = title + " " + post
#     button_style = "background-color: black; color: white; border-radius: 5px;"
#
#     # Si le modèle choisi est présent dans le dictionnaire de fonctions de modèles
#     if model_choice in model_functions:
#
#         # Récupérer la fonction et le nombre de tags associés au modèle choisi
#         model_function = model_functions[model_choice]["function"]
#
#         # Appliquer le modèle choisi à la chaîne d'entrée
#         output = model_function(user_input)
#
#         # Extraire les tags prédits de la sortie
#         if model_choice == "SGDClassifier":
#             tags = list(mlb.inverse_transform(output)[0])
#         else:
#             # tags = output[0]
#             tags = [t[0] for t in output[0]]
#
#         # Impression des tags
#         buttons = "  ".join([f'<button style="{button_style}">{text}</button>' for text in tags])
#         st.markdown(buttons, unsafe_allow_html=True)
