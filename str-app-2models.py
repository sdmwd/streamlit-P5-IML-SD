#!/usr/bin/env python
# coding: utf-8

# # <font color="#114b98">Catégorisez automatiquement des questions</font>

# ## <font color="#114b98">Code final à déployer</font>

# **Stack Overflow** est un site célèbre de questions-réponses liées au développement informatique.

# L'objectif de ce projet est de développer un système de suggestion de tags pour ce site. Celui-ci prendra la forme d’un algorithme de machine learning qui assignera automatiquement plusieurs tags pertinents à une question.

# **Livrable** : Le code final à déployer présenté dans un répertoire et développé progressivement à l’aide d’un logiciel de gestion de versions.

# Lien vers le répertoire : https://github.com/sdmwd/streamlit-P5-IML-SD

# ### Code du fichier principal : streamlit-app-P5-IML-SD.py

# In[4]:


import re
import dill
import nltk
import pickle
import joblib
import sklearn
import html5lib
import streamlit as st
from nltk import pos_tag
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import FunctionTransformer


# In[ ]:


nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# In[ ]:


# path = 'D:/Mega/Z_Simon/5 - WORK/1 - Projets/Projet 5/saved_ressources/'
path = 'ressources/'

# load the saved CountVectorizer
vectorizer_loaded = joblib.load(path + 'countvectorizer.joblib')

# load the saved MultiLabelBinarizer
mlb_loaded = joblib.load(path + 'multilabelbinarizer.joblib')

with open(path + 'stop_words.pkl', 'rb') as f:
    stop_words = pickle.load(f)
    
with open(path + 'pipeline_tags.pkl', 'rb') as file:
    pipeline_tags = dill.load(file)
    
with open(path + 'pipeline_tags_cv.pkl', 'rb') as file:
    pipeline_tags_cv = dill.load(file)


# In[ ]:


def top_words(x):
    pred_count_eval = []
    for i in range(x.shape[0]):
        dense_bow_matrix = x.toarray()
        top_words_indices = dense_bow_matrix[i].argsort()[-5:][::-1]
        top_words_values = dense_bow_matrix[i][top_words_indices]
        top_words_list = []
        for j in range(len(top_words_indices)):
            top_words_list.append((vectorizer_loaded.get_feature_names()[top_words_indices[j]], top_words_values[j]))
        pred_count_eval.append(top_words_list)
    return pred_count_eval


# In[ ]:


st.markdown(
    "<h1 style='margin-top: 0; padding-top: 0;'>Générateur de tags</h1>",
    unsafe_allow_html=True,
)
subtitle = '<p style="font-size: 30px;">Projet 5 - OpenClassrooms Parcours IML</p>'
st.markdown(subtitle, unsafe_allow_html=True)

st.sidebar.header("Choisir un modèle")
model_choice = st.sidebar.selectbox(
    "", 
    ["SGDClassifier", "CountVectorizer"]
)


# In[ ]:


# Main content
user_input = st.text_area("Collez ici un post de Stack Overflow :", height=225)


# In[ ]:


# Model prediction
if model_choice == "SGDClassifier":
    output = pipeline_tags.predict(user_input)
    tags = mlb_loaded.inverse_transform(output)
    st.write('Tags suggérés par SGDClassifier : ' + ', '.join(tags[0]))
else:
    if user_input:
        tags = pipeline_tags_cv.transform(user_input)
        suggested_tags = ', '.join(word for word, _ in tags[0][:5])
    else:
        suggested_tags = ""
    st.write('Tags suggérés par CountVectorizer : ' + suggested_tags)

