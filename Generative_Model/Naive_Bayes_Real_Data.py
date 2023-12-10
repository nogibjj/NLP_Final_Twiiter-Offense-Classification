import re #for regular expressions
import nltk #for text manipulation
import string
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split    
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
import gensim 
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier
from sklearn.metrics import *


train = pd.read_csv('/workspaces/NLP_Final_Twiiter-Offense-Classification/01_Data/02_Processed/train.csv')
test = pd.read_csv('/workspaces/NLP_Final_Twiiter-Offense-Classification/01_Data/02_Processed/test.csv')


# Combining the train and test data for cleaning
combine=pd.concat([train,test],ignore_index=True)


def find_best_alpha(tweets_data):      
    bow_vectorizer = CountVectorizer(ngram_range = (1, 4), max_df=0.90 ,min_df=2 , stop_words='english')
    bow = bow_vectorizer.fit_transform(tweets_data['tokenized'])

    X_train, X_test, y_train, y_test = train_test_split(bow, tweets_data['class'], test_size=0.2, random_state=69)
    best_alpha = 0
    best_score = 0
    for alpha in np.arange(0.1, 1.1, 0.1):
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_alpha = alpha
            best_score = score
    print("Best alpha: {}".format(best_alpha))
    print("Best score: {}".format(best_score))
    return best_alpha


def naive_bayes_model(tweets_data):
   
    bow_vectorizer = CountVectorizer(ngram_range = (1, 4), max_df=0.90 ,min_df=2 , stop_words='english')
    bow = bow_vectorizer.fit_transform(tweets_data['tokenized'])

    X_train, X_test, y_train, y_test = train_test_split(bow, tweets_data['class'],
                                                    test_size=0.2, random_state=69)

    print("X_train_shape : ",X_train.shape)
    print("X_test_shape : ",X_test.shape)
    print("y_train_shape : ",y_train.shape)
    print("y_test_shape : ",y_test.shape)

    best_alpha = find_best_alpha(tweets_data)

    model_naive = MultinomialNB(alpha = best_alpha).fit(X_train, y_train) 
    predicted_naive = model_naive.predict(X_test)

    return predicted_naive, y_test


def conf_mat(tweets_data):
    model_results = naive_bayes_model(tweets_data)
    predicted_naive = model_results[0]
    y_test = model_results[1]
    
    plt.figure(dpi=600)
    mat = confusion_matrix(y_test, predicted_naive)
    sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)

    plt.title('Confusion Matrix for Naive Bayes')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig("confusion_matrix.png")
    return plt.show()


def metrics(tweets_data):
    model_results = naive_bayes_model(tweets_data)
    predicted_naive = model_results[0]
    y_test = model_results[1]

    # Accuracy
    score_naive = accuracy_score(predicted_naive, y_test)
    print("Accuracy with Naive-bayes: ",score_naive)

    # F1 Score
    f1_naive = f1_score(predicted_naive, y_test, average='weighted')
    print("F1 Score with Naive-bayes: ",f1_naive)

    # AUC
    auc_naive = roc_auc_score(predicted_naive, y_test)
    print("AUC with Naive-bayes: ",auc_naive)
    
    return 


metrics(combine)


conf_mat(combine)


naive_bayes_model(combine)


