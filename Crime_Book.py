#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pickle import load
import scipy.io as sio
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['detective', 'series', 'mystery'])

def token_text(text, stopwords = stop_words, lemmatize=True):

    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(text)]
    else:
        tokens = [w for w in word_tokenize(text)]
    token = [w for w in tokens if w not in stopwords]
    
    return token



if __name__ == '__main__':

    data = pd.read_csv('CrimeBookData.csv')
    vect_desc = load(open('vect_desc_single.pkl', 'rb'))
    tfidf_mat_desc = sio.mmread("tfidf_mat_desc_single.mtx")
    vect_books = load(open('vect_books_single.pkl', 'rb'))
    tfidf_mat_books = sio.mmread("tfidf_mat_books_single.mtx")
    

    pat_s = re.compile("\'s")   
    pat_rn = re.compile("\\r\\n") 
    pat_punc = re.compile(r"[^\w\s]") 
    pat_ss = re.compile("\s\s+")
    patterns = [pat_s, pat_rn, pat_punc, pat_ss]


    st.set_page_config(page_title='CrimeBook')

    
    st.title(":blue[CrimeBook]")
    st.subheader(":blue[Mystery book series recommender.]")
    st.write('')
    st.divider()
    st.write('')
    st.header(":blue[**I feel like reading:**]")
    

    test_query = st.text_input(label = "something", label_visibility = "collapsed")
                               
    # select number of books in the series
    st.sidebar.title('Number of books:')
    min_books = st.sidebar.number_input('**At least:**', min_value = 2, max_value = 100, value = 2)
    max_books = st.sidebar.number_input('**At most:**', min_value = 5, max_value = 450, value = 450)

    
    if len(test_query) != 0:
        # clean the input
        test_query = test_query.lower()  
        
        # embed input
        tokens_query = [str(tok) for tok in token_text(test_query)]
        embed_query_desc = vect_desc.transform(tokens_query)
        embed_query_books = vect_books.transform(tokens_query)
        
        # Create list with similarity between query and dataset
        mat_desc = cosine_similarity(embed_query_desc, tfidf_mat_desc)
        mat_books = cosine_similarity(embed_query_books, tfidf_mat_books)
        
        p = 0.4
        if len(mat_desc.shape) > 1:
            cos_sim_desc = np.mean(mat_desc, axis = 0) 
        else: 
            cos_sim_desc = mat_desc
        if len(mat_books.shape) > 1:
            cos_sim_books = np.mean(mat_books, axis = 0) 
        else: 
            cos_sim_books = mat_books
        # mixture of distances
        cos_sim = p*cos_sim_desc + (1 - p)*cos_sim_books
        # sort results
        index = np.argsort(cos_sim)[::-1] 
        # apply cutoff
        mask = (cos_sim[index] > 0.01)
        best_index = index[mask]
        #best_index = index
        
        recomm = data.iloc[best_index]
        recomm = recomm.loc[recomm['series_work_count'] >= min_books, :]
        recomm = recomm.loc[recomm['series_work_count'] <= max_books, :]
        n, dummy = recomm.shape
        if n == 0:
            st.write('No matches. Please provide more info or change filters.')
            topk = 0
        elif n < 5:
            topk = n
        else:
            topk = 5
        
        best_index = recomm.index[:topk]

        # output result
        st.divider()
        st.write('')
        for i in best_index:
            key = data['series_title'].iloc[i]
            link = data['link'].iloc[i]
            st.write(f'### [:blue[{key}]]({link})')
            st.write('**Number of books:** ', str(data['series_work_count'].iloc[i]))
            st.write('**Description**: ', data['series_description'].iloc[i])
            st.write('')
            st.divider()
        if (topk > 0) and (topk < 5):
            st.write('No more matches. Please provide more info or change filters.')
        
        








