# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:23:54 2020

@author: Mhamood

this class contains the manual NLP funtions for cleaning and preprocssing steps 
"""

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords#nltk.download('stopwords')

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import re
#from autocorrect import Speller
def tokenizeSentence(sentence):
    return word_tokenize(sentence)
def removeStopWords(word_tokenize):
#    spell = Speller(lang='en')

    en_stop =stopwords.words('english')
#    return [spell(i) for i in word_tokenize if not spell(i) in en_stop]
    return [i for i in word_tokenize if not i.isdigit() and not i in en_stop ]
    

def stemSentence(sentence):
    token_words=tokenizeSentence(sentence)
    token_words=removeStopWords(token_words)
    porter=PorterStemmer()
    wnl = WordNetLemmatizer()
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(wnl.lemmatize(word)))

    return " ".join(stem_sentence)
def stemdocument(document):
    my_lines_list=" ".join(document.split("\n"))
    stem_sentence=stemSentence(my_lines_list)
    return stem_sentence
def stemSkleanBunch(data):
    for line in range(0,len(data)):
        data[line]=stemdocument(data[line])
    return data

def LemmatizerSentence(sentence):
    token_words=tokenizeSentence(sentence)
    token_words=removeStopWords(token_words)
    wnl = WordNetLemmatizer()
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(wnl.lemmatize(word))

    return " ".join(stem_sentence)
def Lemmatizerdocument(document):
    my_lines_list=" ".join(document.split("\n"))
    stem_sentence=LemmatizerSentence(my_lines_list)
    return stem_sentence
def LemmatizerSkleanBunch(data):
    for line in range(0,len(data)):
        data[line]=Lemmatizerdocument(data[line])
    return data
def clean_text(data):
    """
        text: a string
        
        return: modified initial string
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    for line in range(0,len(data)):
        text=" ".join(data[line].split("\n"))
        text = BeautifulSoup(text, "lxml").text # HTML decoding. BeautifulSoup's text attribute will return a string stripped of any HTML tags and metadata.
        text = text.lower() # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
        text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
        data[line]=text
    return data
def customNLP(data,Lemmatiz=True,stemm=True):
    data=clean_text(data)
    #data=LemmatizerSkleanBunch(data)
    data=stemSkleanBunch(data)
    
    return data
    


def removeEmptyInstances(data,target):
    validdata = []
    for row in data:
        validdata.append(len(row.split()))
    validdata=np.asarray(validdata)
    validIndex = np.where(validdata >0)[0]
    df = pd.DataFrame({'data':data,'target':target})
    df=df.loc[ validIndex,: ]
    data=df['data'].tolist()
    target=df['target'].to_numpy()
    return data,target






#class AFETransformer(TransformerMixin, BaseEstimator):
#    '''A template for a custom transformer.'''
#
#    def __init__(self):
#        super(AFETransformer, self).__init__()
#
#    def fit(self, X, y, **fit_params):
#        
#        X2=X.todense()
#        X2[X2 > 0] = 1
#        X3=[sum(x) for x in zip(*X2)]
#        X2=[]
#        X=X.todense()
#        print(X.shape)
#        self.Classes=np.unique(y, return_index = False)
#        self.vd=np.zeros((len(self.Classes),X.shape[1]))
#        for i in self.Classes:
#            ind = np.where(y == i)[0]
#            self.vd[i,:]=np.matmul(np.log(sum(X[ind,:])+1,axis=0),np.log(X.shape[0])-np.log(sum(X3,axis=0)))
#        X= sparse.csr_matrix(X)
#        return self
#
#    def transform(self, X, **fit_params):
#        X=X.todense()
#        V=np.zeros((X.shape[0],len(self.Classes)))
#        for row in range(0,X.shape[0]):  
#            for i in self.Classes:
#                V[row,i]=np.dot(X[row,:],self.vd[i,:])
#        V= sparse.csr_matrix(V)
#        X=[]
#        return V
    
#from sklearn.base import BaseEstimator, TransformerMixin    
#class DenseTransformer(TransformerMixin):
#
#    def fit(self, X, y=None, **fit_params):
#        return self
#
#    def transform(self, X, y=None, **fit_params):
#        return X.todense()
#    
#    
#class dcdistanceTransformer(TransformerMixin, BaseEstimator):
#    '''A template for a custom transformer.'''
#
#    def __init__(self):
#        super(dcdistanceTransformer, self).__init__()
#
#    def fit(self, X, y, **fit_params):
#        X=X.todense()
#        print(X.shape)
#        self.Classes=np.unique(y, return_index = False)
#        self.vd=np.zeros((len(self.Classes),X.shape[1]))
#        for i in self.Classes:
#            ind = np.where(y == i)[0]
#            self.vd[i,:]=np.mean(X[ind,:],axis = 0)
#        X= sparse.csr_matrix(X)
#        return self
#
#    def transform(self, X, **fit_params):
#        X=X.todense()
#        V=np.zeros((X.shape[0],len(self.Classes)))
#        for row in range(0,X.shape[0]):  
#            for i in self.Classes:
#                V[row,i]=np.linalg.norm(X[row,:]-self.vd[i,:])
#        V= sparse.csr_matrix(V)
#        X=[]
#        return V
