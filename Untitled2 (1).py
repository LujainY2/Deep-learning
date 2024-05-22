#!/usr/bin/env python
# coding: utf-8

# In[5]:


from textblob import TextBlob
import pandas as pd 
import streamlit as st 
import cleantext 


st.header('Sentiment Analysis')
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ',round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ',round(blob.sentiment.subjectivity,2))


    pre = st.text_input('Clean Text: ')
    if pre :
        st.write(cleantext.clean(pre,clean_all=False,extra_space=True,
                                 stopwords=True,lowercase=True,numbers=True,punct=True))

with st.expand('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def score(X):
        blob1 = TextBlob(X)
        return blob1.sentiment.polarity
            
    def analyze(X):
        if X >= 0.5:
            return 'Positive'
        elif X<= -0.5:
            return 'Negative'
        else: 
            return 'Neutral'
        
    if upl:
        df = pd.read_csv(upl)
        df['Score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head())


        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        
        csv = convert_df(df)


        st.download_button(
            label ='Download data as CSV',
            data = csv,
            file_name = 'sentiment.csv',
            mime='text/csv',)
    

