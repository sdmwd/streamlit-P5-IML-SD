#!/usr/bin/env python
# coding: utf-8

# # <font color="#114b98">Catégorisez automatiquement des questions</font>

# ## <font color="#114b98">Code final à déployer</font>

# **Stack Overflow** est un site célèbre de questions-réponses liées au développement informatique.

# L'objectif de ce projet est de développer un système de suggestion de tags pour ce site. Celui-ci prendra la forme d’un algorithme de machine learning qui assignera automatiquement plusieurs tags pertinents à une question.

# **Livrable** : Le code final à déployer présenté dans un répertoire et développé progressivement à l’aide d’un logiciel de gestion de versions.

# In[4]:


import re
import dill
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


# path = 'D:/Mega/Z_Simon/5 - WORK/1 - Projets/Projet 5/saved_ressources/'
path = 'ressources/'


# In[ ]:


# load the saved CountVectorizer
vectorizer_loaded = joblib.load(path + 'countvectorizer.joblib')

# load the saved MultiLabelBinarizer
mlb_loaded = joblib.load(path + 'multilabelbinarizer.joblib')

with open(path + 'stop_words.pkl', 'rb') as f:
    stop_words = pickle.load(f)


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


# Display the value
st.write(output)


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


# In[ ]:



# pydantic==1.8.2
# lxml==4.6.3
# gensim==4.0.1
# beautifulsoup4==4.11.1
# fastapi==0.68.1
# pandas==1.3.2
# scikit_learn==0.24.2
# uvicorn==0.15.0

