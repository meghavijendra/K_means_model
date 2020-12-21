#!/usr/bin/env python
# coding: utf-8

# ## Finding optimum value for number of clusters for k-means clustering

# <p> Here we are trying to find the optimum value of k which needs to be used for the k-means algorithm in order to get the more accurate results</p>
# <p> We can either use this method to get the optimum k values or randomly assign k values to the algorithm and find out which one offer the best accurary</p>

# In[7]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[8]:


#importing the Iris dataset with pandas
df_iris = pd.read_csv('Iris.csv')
x = df_iris.iloc[:, [0, 1, 2, 3]].values


# <p> Here we will implement 'The elbow method' on the Iris dataset. 
#     The elbow method allows us to pick the optimum amount of clusters for classification.</p>

# In[20]:


#Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.annotate('Optimum k value', xy = (3,100), xytext = (3.6, 250),
            arrowprops = dict(facecolor='black', shrink = 0.1),
            horizontalalignment='center', verticalalignment='top')
plt.show()


# <p> The optimum number of clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration. We see that the <b>optimum number of clusters is 3</b> and we can now give this as an input to our kmeans algorithm </p>
