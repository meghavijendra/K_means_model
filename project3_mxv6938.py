#!/usr/bin/env python
# coding: utf-8
"""
Author: Megha Vijendra
ID:1001736938
"""

# Importing the necessary packages
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")


# Check if there are any null values in the dataset and to encode the target column to 0,1,2
def data_preprocessing():
    df_iris = pd.read_csv("iris.csv")
    if df_iris.isnull().values.any():
        df_iris.dropna()
    target_dict = {}
    for i, data in enumerate(df_iris['species'].unique()):
        target_dict[data] = i

    df_iris['species'] = df_iris.apply(lambda x: target_dict[x[4]], axis=1)
    return df_iris


# Shuffling the dataset to get better accuracy and splitting the dataset into train and test
def data_shuffle_split(df_iris):
    df_iris = df_iris.sample(frac=1).reset_index(drop=True)
    total_records = len(df_iris.index)
    split_ratio = 80
    train_len = int(total_records * split_ratio / 100)

    train = df_iris.iloc[:train_len, :4].values
    test = df_iris.iloc[train_len:total_records].values

    return df_iris, train, test


# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def create_k_means_model(K, data):
    n = data.shape[0]
    c = data.shape[1]
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    centroid = np.random.randn(K,c)*std + mean
    EPOCH = 1000
    counter = 0

    centroid_old = np.zeros(centroid.shape)
    centroid_new = deepcopy(centroid)
    clusters = np.zeros(n)
    distances = np.zeros((n,K))
    error = np.linalg.norm(centroid_new - centroid_old)
    while counter != 100 and error!=0 :
        for i in range(K):
            distances[:,i] = np.linalg.norm(data - centroid_new[i], axis=1)
        clusters = np.argmin(distances, axis = 1)

        centroid_old = deepcopy(centroid_new)
        for i in range(K):
            centroid_new[i] = np.mean(data[clusters == i], axis=0)
        counter += 1
        error = np.linalg.norm(centroid_new - centroid_old)
    return centroid_new


# Predicting the clusters of the model created
def predict(model, test):
    distances = []
    for centroid in model:
        distances.append(np.linalg.norm(test - centroid))
    classification = distances.index(min(distances))
    return classification


# getting the accuracy of the model
def getAcc(model, test):
    acc = 0
    total = 0
    for test_data in test:
        if predict(model, test_data[0:4]) == test_data[4]:
            acc += 1
        total += 1
    return acc / total * 100


# As we get different accuracy value because we are choosing random centriod centers we can run the algorithm
# multiple times to find the average overall accuracy of the algorithm.
def main():
    avg_accuracy = 0
    df_iris = data_preprocessing()
    df_iris, train, test = data_shuffle_split(df_iris)
    print("Total number of records in the iris dataset is :", len(df_iris.index))
    print("Splitting the dataset into 80% Train set and 20% Test set")
    print(
        "Enter the value for number of cluster\nYou can find the value of optimum number of clusters using the elbow_method.py file or\nyou can randomly give a value and then check for the one which gives the highest accuracy")
    no_clusters = int(input())
    for x in range(6):
        print("Iteration{}".format(x + 1))
        model = create_k_means_model(3, train)
        acc = getAcc(model, test)
        print("Accuracy = {} %".format(acc))
        avg_accuracy += acc

    avg_accuracy = avg_accuracy / 5
    print("average accuracy =", avg_accuracy)


if __name__ == "__main__":
    main()

