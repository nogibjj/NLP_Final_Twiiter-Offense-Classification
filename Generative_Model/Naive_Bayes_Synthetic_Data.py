
import numpy as np
import pandas as pd
from Naive_Bayes_Real_Data import naive_bayes_model, vocab, combine
import random
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier
from sklearn.metrics import *


def generate_data(model_naive , vocab, sample_size):
    
    p = np.exp(model_naive.feature_log_prob_)

    #Generating non offensive Samples
    r_sentences = []
    for i in range((sample_size // 2)):
        sentence = random.choices (vocab, p[0], k=random.randint(40,60))
        r_sentences.append(" ".join(sentence))
    non_offensive=pd.DataFrame({"Tweets":r_sentences,"labels":0})

    #Generating offensive Samples
    d_sentences = []
    for i in range((sample_size // 2)):
        sentence = random.choices (vocab, p[1], k=random.randint(40,60))
        d_sentences.append(" ".join(sentence))
    offensive = pd.DataFrame({"Tweets":d_sentences,"labels":1})

    return pd.concat([non_offensive, offensive])


synthetic_df = generate_data(naive_bayes_model , vocab, len(combine))


text = synthetic_df['Tweets']
model = CountVectorizer(ngram_range = (2, 2), max_df=0.90 ,min_df=2, stop_words='english')
matrix = model.fit_transform(text).toarray()
df_output = pd.DataFrame(data = matrix, columns = model.get_feature_names())
df_output.T.tail(5) 

bow_vectorizer = CountVectorizer(ngram_range = (2, 2), max_df=0.90 ,min_df=2 , stop_words='english')
bow = bow_vectorizer.fit_transform(synthetic_df['Tweets'])

#synthetic_df=synthetic_df.fillna(0) #replace all null values by 0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bow, synthetic_df['labels'],test_size=0.2, random_state=69)

from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier

model_naive = MultinomialNB().fit(X_train, y_train) 
predicted_naive = model_naive.predict(X_train)

from sklearn.metrics import accuracy_score

score_naive = accuracy_score(predicted_naive, y_test)
print("Accuracy with Naive-bayes: ",score_naive)

from sklearn.metrics import accuracy_score

score_naive = accuracy_score(predicted_naive, y_train)
print("Accuracy with Naive-bayes: ",score_naive)


