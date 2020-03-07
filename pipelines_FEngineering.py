# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:03:55 2020

@author: Mhamood
"""
import transformer as ts
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer,HashingVectorizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
import numpy as np
import string
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

from sklearn.decomposition import PCA, NMF,TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import Normalizer

"""
This file containt simple pipeline function with general purpose:
    case 1: includes the suggested feature selection methoid modefied from the dcdistance in 
    case2: use the input feature reduction method from sklearn or other ML LB
    case 3: no features selection method
    case 4: to stages for feature selection
    parameters: search algorithm parameters
    reductionMethod: feature reduction method
    CV: cross validation method
    model:ML model
"""

# stem function for CountVecrzer
#def stemming_tokenizer(text):
#    stemmer = PorterStemmer()
#    return [stemmer.stem(w) for w in word_tokenize(text)]

# to dense transormer work with pca....(need check)
from sklearn.feature_selection import RFE

import preprocessingNLP as PNLP




def Pipeline_FeatureEngineering(train_data,train_target,parameters=None,CV=10, 
                                reductionMethod=PCA(),reductionType=1,model=LinearSVC(), search="grid"):
    class DenseTransformer(TransformerMixin):

        def fit(self, X, y=None, **fit_params):
            return self

        def transform(self, X, y=None, **fit_params):
            return X.todense()
    class sparseTransformer(TransformerMixin):

        def fit(self, X, y=None, **fit_params):
            return self

        def transform(self, X, y=None, **fit_params):
            return sparse.csr_matrix(X) 
        
    class dcdistanceTransformer(TransformerMixin, BaseEstimator): 
        """
        proposed method modifed from(DCDistance: A Supervised Text Document Feature extraction based on class labels)
        """

        def __init__(self):
            super(dcdistanceTransformer, self).__init__()

        def fit(self, X, y, **fit_params):
            print(X.shape)
            self.Classes=np.unique(y, return_index = False)
            self.vd=np.zeros((len(self.Classes),X.shape[1]))
            for i in self.Classes:
                ind = np.where(y == i)[0]
                self.vd[i,:]=X.tocsr()[ind,:].mean(axis = 0)
               # X= sparse.csr_matrix(X)
            return self

        def transform(self, X, **fit_params):
            V=np.zeros((X.shape[0],len(self.Classes)))
            for row in range(0,X.shape[0]):  
                for i in self.Classes:
                    V[row,i]=np.linalg.norm(X.tocsr()[row,:]-self.vd[i,:])#('Norm',Normalizer(copy=False)),
            V= sparse.csr_matrix(V)
            X=[]
            return V    
        
  
  
        
    if reductionType==1:
        text_clf = Pipeline([('vect', CountVectorizer(max_features=200000,ngram_range=(1,3))),('tfidf', TfidfTransformer()),
                             ('redu',dcdistanceTransformer()),
                             ('Norm',Normalizer(copy=False)),('clf', model)])
    elif reductionType==2:
        text_clf = Pipeline([('vect', CountVectorizer(max_features=200000,ngram_range=(1,3))),('tfidf', TfidfTransformer()),
                             ('redu',reductionMethod),
                             ('Norm',Normalizer(copy=False)),('clf', model)])
    elif reductionType==3:
        text_clf = Pipeline([('vect', CountVectorizer(max_features=200000,ngram_range=(1,2))),('tfidf', TfidfTransformer(use_idf=False)),
                             ('clf', model)])    
    elif reductionType==0:
        text_clf = Pipeline([('vect', CountVectorizer(max_features=200000,ngram_range=(1,3))),('tfidf', TfidfTransformer()),
                             ('redu1',TruncatedSVD(n_components=300)),('Norm1',Normalizer(copy=False)),('redu',reductionMethod),
                             ('Norm2',Normalizer(copy=False)),('clf', model)])
    #global gs_clf

    if search == 'random':
        print("works")
        gs_clf = RandomizedSearchCV(text_clf, parameters, cv=CV, n_jobs=-1, n_iter=5)
    else:
        gs_clf = GridSearchCV(text_clf, parameters, cv=CV, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_target)


        
    return gs_clf





    

"""
untested method

#    class AFETransformer(TransformerMixin, BaseEstimator):
#   
#
#        def __init__(self):
#            super(AFETransformer, self).__init__()
#
#        def fit(self, X, y, **fit_params):
#        
#            #X2=X.todense()
#            X2[X2 > 0] = 1
#            X3=[sum(x) for x in zip(*X2)]
#            X2=[]
#            X=X.todense()
#            print(X.shape)
#            self.Classes=np.unique(y, return_index = False)
#            self.vd=np.zeros((len(self.Classes),X.shape[1]))
#            for i in self.Classes:
#                ind = np.where(y == i)[0]
#                print(np.sum(X[ind,:],axis=0))
#                print(np.log(np.sum(X3,axis=0)))
#                self.vd[i,:]=np.matmul(np.log1p(np.sum(X[ind,:],axis=0)),np.log(X.shape[0])-np.log(X3))
#                X= sparse.csr_matrix(X)
#            return self
#
#        def transform(self, X, **fit_params):
#            X=X.todense()
#            V=np.zeros((X.shape[0],len(self.Classes)))
#            for row in range(0,X.shape[0]):  
#                for i in self.Classes:
#                    V[row,i]=np.dot(X[row,:],self.vd[i,:])
#            V= sparse.csr_matrix(V)
#            X=[]
#            return V  
"""