# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:45:32 2020

@author: Mhamood
Trasformer functions
"""
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
import numpy as np




# stem function for CountVecrzer
def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

# to dense transormer work with pca....(need check)
class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):#,('to_dense', DenseTransformer())
        return X.todense()
    
    
class MyTransformer(TransformerMixin, BaseEstimator):
    '''A template for a custom transformer.'''

    def __init__(self):
        super(MyTransformer, self).__init__()

    def fit(self, X, y, **fit_params):
        X=X.todense()
        print(X.shape)
        self.Classes=np.unique(y, return_index = False)
        self.vd=np.zeros((len(self.Classes),X.shape[1]))
        for i in self.Classes:
            ind = np.where(y == i)[0]
            self.vd[i,:]=sum(X[ind,:])
        X= sparse.csr_matrix(X)
        return self

    def transform(self, X, **fit_params):
        X=X.todense()
        V=np.zeros((X.shape[0],len(self.Classes)))
        for row in range(0,X.shape[0]):  
            for i in self.Classes:
                V[row,i]=np.linalg.norm(X[row,:]-self.vd[i,:])
        V= sparse.csr_matrix(V)
        X=[]
        return V


class AFETransformer(TransformerMixin, BaseEstimator):
    '''A template for a custom transformer.'''

    def __init__(self):
        super(MyTransformer, self).__init__()

    def fit(self, X, y, **fit_params):
        
        X2=X.todense()
        X2[X2 > 0] = 1
        X3=[sum(x) for x in zip(*X2)]
        X2=[]
        X=X.todense()
        print(X.shape)
        self.Classes=np.unique(y, return_index = False)
        self.vd=np.zeros((len(self.Classes),X.shape[1]))
        for i in self.Classes:
            ind = np.where(y == i)[0]
            self.vd[i,:]=np.matmul(np.log(sum(X[ind,:])+1),np.log(X.shape[0])-np.log(X3))
        X= sparse.csr_matrix(X)
        return self

    def transform(self, X, **fit_params):
        X=X.todense()
        V=np.zeros((X.shape[0],len(self.Classes)))
        for row in range(0,X.shape[0]):  
            for i in self.Classes:
                V[row,i]=np.dot(X[row,:],self.vd[i,:])
        V= sparse.csr_matrix(V)
        X=[]
        return V