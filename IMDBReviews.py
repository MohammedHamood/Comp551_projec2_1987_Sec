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

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import load_files

"""
loading test and training data: IMDB Reviews
"""
print("Downloading Dataset ...")
IMDB_train = load_files('C:/Users/oscar/OneDrive - McGill University/Documents/COMP 551/aclImdb/train/',
                        categories=("pos", "neg"), encoding='utf-8')
IMDB_test = load_files('C:/Users/oscar/OneDrive - McGill University/Documents/COMP 551/aclImdb/test/',
                        categories=("pos", "neg"), encoding='utf-8')

print("Dataset Downloaded")

"""
preprocessing
"""
print("PREPROCESSING ...")
IMDB_train.data = PNLP.customNLP(IMDB_train.data)

IMDB_test.data = PNLP.customNLP(IMDB_test.data)

IMDB_train.data, IMDB_train.target = PNLP.removeEmptyInstances(IMDB_train.data, IMDB_train.target)

print("PREPROCESSING DONE!")
"""
logistic regression 
"""
parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (40000,50000),
    # 'vect__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__solver': ('lbfgs'),
    # 'clf__penalty': ('l2','none'),
    # 'clf__max_iter': (300,200),
    # 'clf__C': (5,),
}
#max_iter=300,tol=.0001,C=5
LR_base = BaseLine.Pipeline_FeatureEngineering(IMDB_train.data, IMDB_train.target,
                                               parameters=parameters, CV=10,
                                               reductionMethod=TruncatedSVD(n_components=100),
                                               reductionType=3, model=LogisticRegression())
LR_base_predict = LR_base.predict(IMDB_test.data)
print("Training Accuracy: " + str(LR_base.best_score_))
print("Testing Accuracy: " + str(np.mean(LR_base_predict == IMDB_test.target)))
print(LR_base.best_estimator_)
#print(LR_base.score(LR_base_predict, IMDB_test.target))

"""
SVM 
"""
parameters = {
    # 'vect__max_df': (0.99, 1),
    # 'vect__min_df': (0.01, 0),
    # 'vect__max_features': (5000,10000,30000),
    # 'vect__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__solver': ('lbfgs'),
    # 'clf__penalty': ('l2','none'),
    # 'clf__max_iter': (1000,2000),
    'clf__C': (1, .5),
}

#SVM_base = BaseLine.Pipeline_FeatureEngineering(train_data.data, train_data.target
                                                #, parameters=parameters, CV=3,
                                                #reductionMethod=TruncatedSVD(n_components=1000)
                                                #, reductionType=2, model=LinearSVC(max_iter=3000))
#SVM_base_pridect = SVM_base.predict(test_data.data)
#print(np.mean(SVM_base_pridect == test_data.target))

#print(SVM_base.best_estimator_)
#print(SVM_base.best_score_)
#print(SVM_base.score(SVM_base_pridect, test_data.target))

"""
 DecisionTreeClassifie train
"""
parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 1000,5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__criterion': ('gini', 'entropy'),
    'clf__ccp_alpha': (0, .05, .1, .2, .3, .4),
    'clf__max_depth': (None, 5, 7, 8, 10, 12, 20, 100),
    'clf__min_samples_split': (2, 3, 4, 8, 10, 50, 100),
    'clf__min_samples_leaf': (1, 2, 3, 4, 8, 10, 50, 100),
    'clf__max_features': (None, 'auto', 'sqrt', 'log2'),
    'clf__max_leaf_nodes': (None, 20, 30, 40, 80),
}

#DT_base = BaseLine.Pipeline_FeatureEngineering(train_data.data, train_data.target,
                                               #parameters=parameters, CV=10,
                                               #reductionMethod=TruncatedSVD(n_components=100),
                                               #reductionType=3, model=DecisionTreeClassifier())

"""
 DecisionTreeClassifie test
"""

#DT_base_pridect = DT_base.predict(test_data.data)
#print(np.mean(DT_base_pridect == test_data.target))
#print(DT_base.best_estimator_)
#print(DT_base.best_score_)
#print(DT_base.score(DT_base_pridect, test_data.target))

parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 1000,5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__criterion': ('gini', 'entropy'),
    'clf__ccp_alpha': (0, .05, .1, .2, .3, .4),
    'clf__max_depth': (None, 5, 7, 8, 10, 12, 20, 100),
    'clf__min_samples_split': (2, 3, 4, 8, 10, 50, 100),
    'clf__min_samples_leaf': (1, 2, 3, 4, 8, 10, 50, 100),
    'clf__max_features': (None, 'auto', 'sqrt', 'log2'),
    'clf__max_leaf_nodes': (None, 20, 30, 40, 80),
    'clf__n_estimators': (8, 10, 20, 40, 100, 200, 500),
}

#RF_base = BaseLine.Pipeline_FeatureEngineering(train_data.data, train_data.target,
                                               #parameters=parameters, CV=10,
                                               #reductionMethod=TruncatedSVD(n_components=100),
                                               #reductionType=3, model=RandomForestClassifier())
"""
 RandomForestClassifier  test
"""
#RF_base_pridect = RF_base.predict(test_data.data)
#print(np.mean(RF_base_pridect == test_data.target))
#print(RF_base.best_estimator_)
#print(RF_base.best_score_)
#print(RF_base.score(RF_base_pridect, test_data.target))

##
##
##
##
##
##


"""
 AdaBoostClassifier train with RF 
"""

parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 1000,5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__solver': ('lbfgs'),
    'clf__n_estimators': (8, 10, 20, 30, 50, 100, 200),
    'clf__base_estimator__criterion': ('gini', 'entropy'),
    'clf__base_estimator__ccp_alpha': (0, .05, .1, .2, .3, .4),
    'clf__base_estimator__max_depth': (None, 5, 7, 8, 10, 12, 20, 100),
    'clf__base_estimator__min_samples_split': (2, 3, 4, 8, 10, 50, 100),
    'clf__base_estimator__min_samples_leaf': (1, 2, 3, 4, 8, 10, 50, 100),
    'clf__base_estimator__max_features': (None, 'auto', 'sqrt', 'log2'),
    'clf__base_estimator__max_leaf_nodes': (None, 20, 30, 40, 80),
}

#ada_RF_base = BaseLine.Pipeline_FeatureEngineering(train_data.data, train_data.target,
                                                   #parameters=parameters, CV=10,
                                                   #reductionMethod=TruncatedSVD(n_components=100),
                                                   #reductionType=3, model=AdaBoostClassifier())

"""
 AdaBoostClassifier  test with RF
"""
#ada_RF_base_pridect = ada_RF_base.predict(test_data.data)
#print(np.mean(RF_base_pridect == test_data.target))
#print(ada_RF_base.best_estimator_)
#print(ada_RF_base.best_score_)
#print(ada_RF_base.score(ada_RF_base_pridect, test_data.target))

"""
 AdaBoostClassifier train with SVM 
"""
parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 1000,5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__solver': ('lbfgs'),
    'clf__base_estimator': LinearSVC(),
    'clf__n_estimators': (8, 10, 20, 30, 50, 100, 200),
    'clf__base_estimator__max_iter': (100, 200, 300, 400, 500, 1000, 2000),
    'clf__base_estimator__C': (100, 50, 10, 5, 2, 1, .8, .5, .1, .01),
}

#ada_SVM_base = BaseLine.Pipeline_FeatureEngineering(train_data.data, train_data.target,
                                                    #parameters=parameters, CV=10,
                                                    #reductionMethod=TruncatedSVD(n_components=100),
                                                    #reductionType=3, model=AdaBoostClassifier())

"""
 AdaBoostClassifier  test with SVM
"""
#ada_SVM_base_pridect = ada_SVM_base.predict(test_data.data)
#print(np.mean(ada_SVM_base_pridect == test_data.target))
#print(ada_SVM_base.best_estimator_)
#print(ada_SVM_base.best_score_)
#print(ada_SVM_base.score(ada_SVM_base_pridect, test_data.target))

"""
 AdaBoostClassifier train with LR 
"""
parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 1000,5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__solver': ('lbfgs'),
    'clf__base_estimator': LinearSVC(),
    'clf__n_estimators': (8, 10, 20, 30, 50, 100, 200),
    'clf__base_estimator__max_iter': (100, 200, 300, 400, 500, 1000, 2000),
    'clf__base_estimator__C': (100, 50, 10, 5, 2, 1, .8, .5, .1, .01),
}

#ada_LR_base = BaseLine.Pipeline_FeatureEngineering(train_data.data, train_data.target,
                                                   #parameters=parameters, CV=10,
                                                   #reductionMethod=TruncatedSVD(n_components=100),
                                                   #reductionType=3, model=AdaBoostClassifier())

"""
 AdaBoostClassifier  test with LR
"""
#ada_LR_base_pridect = ada_LR_base.predict(test_data.data)
#print(np.mean(ada_SVM_base_pridect == test_data.target))
#print(ada_LR_base.best_estimator_)
#print(ada_LR_base.best_score_)
#print(ada_LR_base.score(ada_LR_base_pridect, test_data.target))

"""
eXTRA: NAIVE BASE 
"""
parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (40000, 50000),
    # 'vect__ngram_range': ((1, 1), (1, 2),(1, 3)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__solver': ('lbfgs'),
    # 'clf__penalty': ('l2','none'),
    # 'clf__max_iter': (300,200),
    # 'clf__C': (5,),
}

#NB_base = BaseLine.Pipeline_FeatureEngineering(train_data.data, train_data.target,
                                               #parameters=parameters, CV=10,
                                               #reductionMethod=TruncatedSVD(n_components=100),
                                               #reductionType=3, model=MultinomialNB())
#NB_base_pridect = NB_base.predict(test_data.data)
#print(np.mean(NB_base_pridect == test_data.target))
#print(NB_base.best_estimator_)
#print(NB_base.best_score_)
#print(NB_base.score(NB_base_pridect, test_data.target))



