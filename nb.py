import numpy as np
import pandas as pd
import math 

from google.colab import drive
drive.mount('/content/drive')

df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/dmw8-1.csv',delimiter='\t')
df=df.set_index('Day')

def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    print('classes', classes)
    prior = {}
    for i in classes:
        print(i)
        prior[i]=(len(df[df[Y]==i])/len(df))
    return prior


def prob_x_given_y(df, feature_name, feature_val, Y, label):
    df = df[df[Y] == label]
    count_y = df[Y].count()
    count_xy = df[df[feature_name] == feature_val][Y].count()
    p_x_given_y = (count_xy + 1) / (count_y + len(df[feature_name].unique()))
    return p_x_given_y


def naive_bayes_classifier(df, X, Y):
    # Calculate class priors
    prior = calculate_prior(df, Y)
    print('prior',prior)

    # Calculate likelihoods for each class and feature combination
    likelihoods = {}
    for label in sorted(df[Y].unique()):
        likelihoods[label] = {}
        for feature_name in X:
            for feature_val in df[feature_name].unique():
                likelihoods[label][(feature_name, feature_val)] = prob_x_given_y(df, feature_name, feature_val, Y, label)
    
    # Classify new instances
    def classify(instance):
        scores = {}
        for label in sorted(df[Y].unique()):
          #  print('label',label,prior)
            score = prior[label]
            for feature_name in X:
                feature_val = instance[feature_name]
                score *= likelihoods[label][(feature_name, feature_val)]
            scores[label]=(score)
        # this scores has numerator now get actual prob
        probs={}
        pr=sum(list(scores.values()))
        for key,val in scores.items():
          probs[key]=val/pr
        print('scores are',scores)
        print('probablities are',probs)
        
        return
    q1={'Outlook':'Rain','Temperature':'Hot','Humidity':'Normal','Wind':'Weak'}
    classify(q1)

X=list(df.columns)[:-1]
Y=str(list(df.columns)[-1])
print(X,Y)
naive_bayes_classifier(df,X,Y)