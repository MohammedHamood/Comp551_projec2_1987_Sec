# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:03:55 2020

@author: Mhamood
"""
#import transformer as ts
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
#from nltk.stem import PorterStemmer
#from nltk import word_tokenize


from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
import numpy as np
#import string
from sklearn.pipeline import Pipeline
#from nltk.corpus import stopwords

from sklearn.decomposition import PCA,TruncatedSVD
#from sklearn.feature_selection import SelectKBest, chi2

#from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer

from imblearn.over_sampling import ADASYN


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


"""
THIS METHOD ALSO INCLUDES ('samp',ADASYN(random_state=42)) STEP TO INMPROVE THE PREDICTABILITY
"""



def Pipeline_FeatureEngineering(train_data,train_target,parameters=None,CV=10, 
                                reductionMethod=PCA(),reductionType=1,model=LinearSVC()):
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
            self.alpha=20
            self.Classes=np.unique(y, return_index = False)
            self.split=np.linspace(0, X.shape[1],self.alpha,endpoint = False,  dtype=int)
            self.vd=np.zeros((len(self.Classes),X.shape[1]))
            self.vd2=np.zeros((len(self.Classes),X.shape[1]))
            for i in self.Classes:
                ind = np.where(y == i)[0]
                temp=np.zeros((1,X.shape[1]))
                for feat in range(0,len(self.split)):
                    ind1=self.split[feat]              
                    ind2=self.split[feat]+self.split[1]
                    temp=X.tocsr()[ind,ind1:ind2].sum(axis = 0)
                    self.vd2[int(i),ind1:ind2]=temp     
            tfidf_transformer = TfidfTransformer()
            self.vd2 = tfidf_transformer.fit_transform(self.vd2) 
            self.vd2=self.vd2.todense()   
            return self

        def transform(self, X, **fit_params):
            V= np.zeros((X.shape[0],len(self.Classes)))

            
            for i in self.Classes:
                starting=0
                ending=200
                while starting<X.shape[0]-2:
                    if ending>X.shape[0]:
                        ending=X.shape[0]
                    temp=np.linalg.norm(X.tocsr()[starting:ending,:]-self.vd2[int(i),:],axis=1,ord=2)
                    V[starting:ending,int(i)]= temp[:] #('Norm',Normalizer(copy=False)),
                    starting=ending
                    ending=ending+200
            V= sparse.csr_matrix(V)
            X=[]
            return V         

        
    if reductionType==1:
        text_clf = Pipeline([('vect', CountVectorizer(max_features=200000,ngram_range=(1,3))),('samp',ADASYN(random_state=42)),('tfidf', TfidfTransformer()),
                             ('redu',dcdistanceTransformer()),('clf', model)])
    elif reductionType==2:
        text_clf = Pipeline([('vect', CountVectorizer(max_features=200000,ngram_range=(1,3))),('samp',ADASYN(random_state=42)),('tfidf', TfidfTransformer()),
                             ('redu',reductionMethod),
                             ('Norm',Normalizer(copy=False)),('clf', model)])
    elif reductionType==3:
        text_clf = Pipeline([('vect', CountVectorizer(max_features=200000,ngram_range=(1,3))),('samp',ADASYN(random_state=42)),('tfidf', TfidfTransformer()),
                             ('clf', model)])    
    elif reductionType==0:
        text_clf = Pipeline([('vect', CountVectorizer(max_features=200000,ngram_range=(1,3))),('samp',ADASYN(random_state=42)),('tfidf', TfidfTransformer()),
                             ('redu1',TruncatedSVD(n_components=300)),('Norm1',Normalizer(copy=False)),('redu',reductionMethod),
                             ('Norm2',Normalizer(copy=False)),('clf', model)])
    gs_clf = GridSearchCV(text_clf, parameters, cv=CV, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_target)
    return gs_clf

