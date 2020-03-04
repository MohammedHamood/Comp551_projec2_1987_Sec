# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:33:01 2020

@author: Mhamood
"""
import pandas as pd
import nltk
nltk.download('stopwords')

import nltk
nltk.download('punkt')
import nltk
nltk.download('wordnet')

import pipelines_FEngineering as BaseLine
from sklearn.datasets import fetch_20newsgroups
import preprocessingNLP as PNLP
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA, NMF,TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2

train_data = fetch_20newsgroups(subset='train',shuffle=True, random_state=42,remove=['headers', 'footers', 'quotes'])
train_data.data=PNLP.customNLP(train_data.data)
test_data = fetch_20newsgroups(subset='test',shuffle=True, random_state=42,remove=['headers', 'footers', 'quotes'])
test_data.data=PNLP.customNLP(test_data.data)


validdata = []
for row in train_data.data:
    validdata.append(len(row.split()))
validdata=np.asarray(validdata)
validIndex = np.where(validdata >0)[0]
print(validIndex.shape)
df = pd.DataFrame({'data':train_data.data,'target':train_data.target})
df=df.loc[ validIndex,: ]
train_data.data=df['data'].tolist()
train_data.target=df['target'].to_numpy()

## logistic regression train
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (40000,50000),
    #'vect__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__solver': ('lbfgs'),
    #'clf__penalty': ('l2','none'),
    #'clf__max_iter': (300,200),
    #'clf__C': (5,),
}


LR_base=BaseLine.Pipeline_FeatureEngineering(train_data.data,train_data.target,
                                             parameters=parameters,CV=10,
                                             reductionMethod=TruncatedSVD(n_components=100),
                                             reductionType=3,model=LogisticRegression(max_iter=300,tol=.0001,C=5))
LR_base_pridect=LR_base.predict(test_data.data)
print(np.mean(LR_base_pridect == test_data.target))
LR_base.best_estimator_
LR_base.best_score_

# SVM regression train
parameters = {
    #'vect__max_df': (0.99, 1),
    #'vect__min_df': (0.01, 0),
    #'vect__max_features': (5000,10000,30000),
    'vect__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__solver': ('lbfgs'),
    #'clf__penalty': ('l2','none'),
    'clf__max_iter': (100,200,300,1000,2000,3000),
    'clf__C': (100, 50,10, 5, 2,1,.8,.5,.1,.01),
}


SVM_base=BaseLine.Pipeline_FeatureEngineering(train_data.data,train_data.target
                                              ,parameters=parameters,CV=10,
                                              reductionMethod=TruncatedSVD(n_components=100)
                                              ,reductionType=3,model=LinearSVC())
SVM_base_pridect=SVM_base.predict(test_data.data)
print(np.mean(SVM_base_pridect == test_data.target))





#
#
## DecisionTreeClassifie train
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 1000,5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2),(1, 3)), 
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__criterion': ('gini','entropy'),
    'clf__ccp_alpha': (0,.05,.1,.2,.3,.4),
    'clf__max_depth': (None,5,7,8,10,12,20,100),
    'clf__min_samples_split': (2,3,4,8,10,50,100),
    'clf__min_samples_leaf': (1,2,3,4,8,10,50,100),
    'clf__max_features': (None,'auto', 'sqrt', 'log2'),
    'clf__max_leaf_nodes': (None,20,30,40,80),
    }

DT_base=BaseLine.Pipeline_FeatureEngineering(train_data.data,train_data.target,
                                             parameters=parameters,CV=10,
                                             reductionMethod=TruncatedSVD(n_components=100),
                                             reductionType=3,model=DecisionTreeClassifier())
#
#
## RandomForestClassifier train
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 1000,5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2),(1, 3)), 
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__criterion': ('gini','entropy'),
    'clf__ccp_alpha': (0,.05,.1,.2,.3,.4),
    'clf__max_depth': (None,5,7,8,10,12,20,100),
    'clf__min_samples_split': (2,3,4,8,10,50,100),
    'clf__min_samples_leaf': (1,2,3,4,8,10,50,100),
    'clf__max_features': (None,'auto', 'sqrt', 'log2'),
    'clf__max_leaf_nodes': (None,20,30,40,80),
    'clf__n_estimators': (8,10,20,40,100,200,500),
    }

RF_base=BaseLine.Pipeline_FeatureEngineering(train_data.data,train_data.target,
                                             parameters=parameters,CV=10,
                                             reductionMethod=TruncatedSVD(n_components=100),
                                             reductionType=3,model=RandomForestClassifier())
RF_base_pridect=RF_base.predict(test_data.data)
print(np.mean(RF_base_pridect == test_data.target))
#
#
#
#
#
#
## AdaBoostClassifier train
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 1000,5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2),(1, 3)), 
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__solver': ('lbfgs'),
    'clf__n_estimators': (8,10,20,30,50,100,200),
    'clf__base_estimator__criterion': ('gini','entropy'),
    'clf__base_estimator__ccp_alpha': (0,.05,.1,.2,.3,.4),
    'clf__base_estimator__max_depth': (None,5,7,8,10,12,20,100),
    'clf__base_estimator__min_samples_split': (2,3,4,8,10,50,100),
    'clf__base_estimator__min_samples_leaf': (1,2,3,4,8,10,50,100),
    'clf__base_estimator__max_features': (None,'auto', 'sqrt', 'log2'),
    'clf__base_estimator__max_leaf_nodes': (None,20,30,40,80),
    }

ada_RF_base=BaseLine.Pipeline_FeatureEngineering(train_data.data,train_data.target,
                                             parameters=parameters,CV=10,
                                             reductionMethod=TruncatedSVD(n_components=100),
                                             reductionType=3,model=AdaBoostClassifier())
ada_RF_base_pridect=ada_RF_base.predict(test_data.data)
print(np.mean(ada_RF_base_pridect == test_data.target))
#
#
#
## AdaBoostClassifier train
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 1000,5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2),(1, 3)), 
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__solver': ('lbfgs'),
    'clf__base_estimator':(LogisticRegression(),LinearSVC()),
    'clf__n_estimators': (8,10,20,30,50,100,200),
    'clf__base_estimator__max_iter': (100,200,300,400,500,1000,2000),
    'clf__base_estimator__C': (100, 50,10, 5, 2,1,.8,.5,.1,.01),
    }

ada_LR_base=BaseLine.Pipeline_FeatureEngineering(train_data.data,train_data.target,
                                             parameters=parameters,CV=10,
                                             reductionMethod=TruncatedSVD(n_components=100),
                                             reductionType=3,model=AdaBoostClassifier())
ada_LR_base_pridect=ada_LR_base.predict(test_data.data)
print(np.mean(ada_LR_base_pridect == test_data.target))
#

## AdaBoostClassifier train
#parameters = {
#    #'vect__max_df': (0.5, 0.75, 1.0),
#    'vect__max_features': (None, 1000,5000, 10000, 50000),
#    'vect__ngram_range': ((1, 1), (1, 2),(1, 3)), 
#    #'tfidf__use_idf': (True, False),
#    #'tfidf__norm': ('l1', 'l2'),
#    #'clf__solver': ('lbfgs'),
#    'clf__base_estimator':(LogisticRegression(),LinearSVC()),
#    'clf__n_estimators': (8,10,20,30,50,100,200),
#    'clf__base_estimator__max_iter': (100,200,300,400,500),
#    'clf__base_estimator__C': (100, 50,10, 5, 2,1,.8,.5,.1,.01),
#    }
#
#ada_SVM_base=BaseLine.Pipeline_FeatureEngineering(train_data.data,train_data.target,
#                                             parameters=parameters,CV=10,
#                                             reductionMethod=TruncatedSVD(n_components=100),
#                                             reductionType=3,model=DecisionTreeClassifier())
#ada_SVM_base_pridect=ada_SVM_base.predict(test_data.data)
#print(np.mean(ada_SVM_base_pridect == test_data.target))