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

with open(path + 'lda.pkl', 'rb') as file:
    lda = dill.load(file)

# Définir un dictionnaire de fonctions de modèles et de leurs paramètres associés
model_functions_supervised = {
    "SGDClassifier": {"function": pipelines["SGDClassifier"].predict, "tag_transform": lambda output: list(mlb.inverse_transform(output)[0])},
}

model_functions_unsupervised = {
    "CountVectorizer": {"function": pipelines["CountVectorizer"].transform, "tag_transform": lambda output: list(t[0] for t in output[0])},
    "TFIDFVectorizer": {"function": pipelines["TFIDFVectorizer"].transform, "tag_transform": lambda output: list(t[0] for t in output[0])},
    "LDA": {"function": pipelines["LDA"].transform, "tag_transform": lambda output: list(t[0] for t in output[0])},
}

# Définition de l'interface utilisateur
st.markdown(
    "<h1 style='margin-top: 0; padding-top: 0;'>Générateur de tags</h1>",
    unsafe_allow_html=True)
subtitle = '<p style="font-size: 30px;">Projet 5 - OpenClassrooms Parcours IML</p>'
st.markdown(subtitle, unsafe_allow_html=True)


# Sélection du modèle à utiliser
st.sidebar.header("Choisir un modèle")

with st.sidebar.container():
    choice = st.selectbox(" ", [" ", "Approche supervisée", "Approche non supervisée"])
    model_choice = None

    if choice == "Approche supervisée":
        with st.sidebar.container():
            model_choice = st.selectbox(" ", model_functions_supervised.keys())

    if choice == "Approche non supervisée":
        with st.sidebar.container():
            model_choice = st.selectbox(" ", model_functions_unsupervised.keys())


# Saisie du titre et du texte à utiliser
title = st.text_input("Collez ici votre titre :")
post = st.text_area("Collez ici votre texte :", height=250)


# Si aucune approche ni aucun modèle ne sont sélectionnés, afficher un message d'erreur
if model_choice is None:
    st.error("Merci de sélectionner un modèle.")
# 
# elif model_choice == "  ":
#     st.error("Merci de sélectionner un modèle.")

else:

    # Génération des tags si l'utilisateur a cliqué sur le bouton et a fourni des données
    if st.button("Generate Tags") and title and post and (model_choice is not None or ""):

        # Concaténer le titre et le message en une seule chaîne
        user_input = title + " " + post
        button_style = "background-color: black; color: white; border-radius: 5px;"

        # Récupérer la fonction pour les modèles supervisés
        if model_choice in model_functions_supervised:
            model_function = model_functions_supervised[model_choice]["function"]
            tag_transform = model_functions_supervised[model_choice]["tag_transform"]

        # Récupérer la fonctionpour les modèles non supervisés
        elif model_choice in model_functions_unsupervised:
            model_function = model_functions_unsupervised[model_choice]["function"]
            tag_transform = model_functions_unsupervised[model_choice]["tag_transform"]

        # Appliquer le modèle choisi à la chaîne d'entrée
        output = model_function(user_input)

        # Extraire les tags prédits de la sortie
        tags = tag_transform(output)

        # Impression des tags
        buttons = "  ".join([f'<button style="{button_style}">{text}</button>' for text in tags])
        st.markdown(buttons, unsafe_allow_html=True)
