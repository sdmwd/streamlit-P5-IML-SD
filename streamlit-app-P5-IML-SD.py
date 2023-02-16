#!/usr/bin/env python
# coding: utf-8

# # <font color="#114b98">Catégorisez automatiquement des questions</font>

# ## <font color="#114b98">Code final à déployer</font>

# **Stack Overflow** est un site célèbre de questions-réponses liées au développement informatique.

# L'objectif de ce projet est de développer un système de suggestion de tags pour ce site. Celui-ci prendra la forme d’un algorithme de machine learning qui assignera automatiquement plusieurs tags pertinents à une question.

# **Livrable** : Le code final à déployer présenté dans un répertoire et développé progressivement à l’aide d’un logiciel de gestion de versions.

# In[ ]:


import os
import dill
import streamlit as st
import re
import spacy
import joblib
import html5lib
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
spacy.load('en_core_web_sm')
# pip install streamlit


# In[ ]:


# path = 'D:/Mega/Z_Simon/5 - WORK/1 - Projets/Projet 5/saved_ressources/'
path = 'ressources/'


# In[ ]:


# load the saved CountVectorizer
vectorizer_loaded = joblib.load(path + 'countvectorizer.joblib')

# # load the saved classifier
# clf_loaded = joblib.load('D:/Mega/Z_Simon/5 - WORK/1 - Projets/Projet 5/saved_ressources/sgdc_classifier.pkl')

# load the saved MultiLabelBinarizer
mlb_loaded = joblib.load(path + 'multilabelbinarizer.joblib')


# In[ ]:


st.markdown(
    "<h1 style='margin-top: 0; padding-top: 0;'>Générateur de tags</h1>",
    unsafe_allow_html=True,
)


# In[ ]:


# st.title("Générateur de tags")
subtitle = '<p style="font-size: 30px;">Projet 5 - OpenClassrooms Parcours IML</p>'
st.markdown(subtitle, unsafe_allow_html=True)


# In[ ]:


user_input = st.text_area("Collez ici un post de Stack Overflow:", height=150)


# In[ ]:


with open(path + 'pipeline_tags.pkl', 'rb') as file:
    pipeline_tags = dill.load(file)


# In[ ]:


output = pipeline_tags.predict(user_input)


# In[ ]:


tags = mlb_loaded.inverse_transform(output)


# In[ ]:


# Display the value
st.write('Tags suggérés :')
for tag in tags[0]:
    st.write('- ' + tag)


# In[ ]:


# streamlit run D:/Mega/Z_Simon/5 - WORK/1 - Projets/Projet 5/saved_ressources/Durand_Simon_3_code_012023.py


# In[ ]:


# streamlit run C:\Users\simon\Downloads\Durand_Simon_3_code_012023.py   

