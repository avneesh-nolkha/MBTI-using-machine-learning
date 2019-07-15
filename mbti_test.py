#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:23:05 2019

@author: avneeshnolkha
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from io import StringIO

import re
import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk import word_tokenize

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from xgboost import XGBClassifier

#Importing the Dataset
data = pd.read_csv("mbti_1.csv")

#Giving unique id to each personality type
data['personality_id']=data['type'].factorize()[0]
mbti_types=data['type'].unique().tolist()

#Plotting graph to check distribution of personality types
fig=plt.figure(figsize=(8,8))
data.groupby('type').posts.count().plot.bar(ylim=0)
plt.show()

#DataPrepocessing
"""Removing '|||' from the posts"""
data['seperated_post'] = data['posts'].apply(lambda x: x.strip().split("|||"))
data['id'] = data.index

df = pd.DataFrame(data['seperated_post'].tolist(), index=data['id']).stack().reset_index(level=1, drop=True).reset_index(name='idposts')

df=df.join(data.set_index('id'), on='id', how = 'left')

df =df.drop(columns=['posts','seperated_post'])

#Using RE to clean posts
def clean_text(text):
    result = re.sub(r'http[^\s]*', '',text)
    result = re.sub('[0-9]+','', result).lower()
    result = re.sub('@[a-z0-9]+', '', result)
    return re.sub('[%s]*' % string.punctuation, '',result)

def deEmojify(text):
    return text.encode('ascii', 'ignore').decode('ascii')

df['idposts']=df['idposts'].apply(clean_text)
df['idposts']=df['idposts'].apply(deEmojify)
cleaned_df = df.groupby('id')['idposts'].apply(list).reset_index()

data['clean_post'] = cleaned_df['idposts'].apply(lambda x: ' '.join(x))

unique_type_list = [x.lower() for x in mbti_types]

#df=df.dropna(axis=0,how='any')
stemmer = PorterStemmer()

list1=stopwords.words('english')

unique_type_list= list1 + unique_type_list

vectorizer = CountVectorizer(stop_words = unique_type_list,
                            max_features=1500,
                            analyzer="word",
                            )

corpus = data['clean_post'].values.astype('U').reshape(1,-1).tolist()[0]
vectorizer.fit(corpus)
features = vectorizer.fit_transform(corpus).toarray()

"""Since it will be very difficult and computation intensive for us to run a classification task on such a big sparse matrix and classify it for 16 labels, we will divide the labels such that there are 4 types and each type has 2 classes. Thus we will now have to perform classification 4 times , each time for a different type and we will combine the final classes
Eg.: 
  Type 1  Introversion (I) – Extroversion (E)
  Type 2  Intuition (N) – Sensing (S)
  Type 3  Thinking (T) – Feeling (F)
  Type 4  Judging (J) – Perceiving (P)
  
  INFJ - Introvert Intution Feeling Judging
  ENFP - Entrovert Intuition Feeling Perciving
  Thus we perform perform classification on these 4 types and combine the final result"""
  
data['type1'] = data['type'].apply(lambda x: 1 if x[0] == 'E' else 0)
data['type2'] = data['type'].apply(lambda x: 1 if x[1] == 'S' else 0)
data['type3'] = data['type'].apply(lambda x: 1 if x[2] == 'T' else 0)
data['type4'] = data['type'].apply(lambda x: 1 if x[3] == 'J' else 0)

all_words = vectorizer.get_feature_names()
n_words = len(all_words)
df = pd.DataFrame.from_dict({w: features[:, i] for i, w in enumerate(all_words)})

scaler = MinMaxScaler()
features=scaler.fit_transform(features)



"""X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.5, random_state = 0)

#Using SVM
classifier = SVC(kernel='poly',degree=128,gamma='scale',C=0.1)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
#Making DeepLearningModel
#Initialising the ANN
classifier = Sequential()


# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(output_dim=64, kernel_initializer='uniform', activation= 'relu',input_dim=1500))
classifier.add(Dropout(p=0.15))

classifier.add(Dense(output_dim=1024, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dropout(p=0.15))

classifier.add(Dense(output_dim=512, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dropout(p=0.15))

classifier.add(Dense(output_dim=128, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dropout(p=0.15))

classifier.add(Dense(output_dim=64, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dropout(p=0.15))

classifier.add(Dense(output_dim=32, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dropout(p=0.15))

classifier.add(Dense(output_dim=16, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dropout(p=0.15))

classifier.add(Dense(output_dim=4, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dropout(p=0.15))
#Adding the final/output layer
classifier.add(Dense(output_dim=1, kernel_initializer='uniform', activation= 'softmax'))

#Compiling the ANN
classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Fitting the ANN to training set
classifier.fit(X_train, y_train,batch_size=32, epochs=100)"""


    

def classifier(keyword):
    y = df[keyword].values
    X_train, X_test, y_train, y_test = train_test_split(features, y, stratify=y)
    classifier = XGBClassifier()
    flassifier.fit(X_train, y_train, 
                     early_stopping_rounds = 10, 
                     eval_metric="logloss", 
                     eval_set=[(X_test, y_test)], verbose=False)
    print("%s:" % keyword, sum(y)/len(y))
    print("Accuracy %s" % keyword, accuracy_score(y_test, classifier.predict(X_test)))
    return classifier






















