#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score


# In[2]:


df = pd.read_csv("Traffic.csv")

df.drop(columns=['Time', 'Day of the week', "Traffic Situation"], axis=1,inplace=True )
df.describe()


# In[3]:


models = []
score1 = []
score2 = []
for i in range (2,6):
    model = KMeans(n_clusters=i, random_state=250, init='k-means++', n_init='auto').fit(df)
    models.append(model)
    score1.append(model.inertia_)
    score2.append(silhouette_score(df, model.labels_))


# In[4]:


plt.grid()
plt.plot(np.arange(2,6), score1, marker = 'o')
plt.show()


# In[5]:


plt.grid()
plt.plot(np.arange(2,6), score2, marker = 'o')
plt.show()


# In[6]:


model1 = KMeans(n_clusters=2, random_state=100, init='k-means++', n_init='auto')
model1.fit(df)

labels = model1.labels_
df['Cluster'] = labels
fig = go.Figure(data=[go.Scatter(x=df['CarCount'],y=df['BikeCount'], mode='markers', marker_color=df['Cluster'], marker_size = 4)])
fig.show()


# In[7]:


from sklearn.cluster import AgglomerativeClustering

model2 = AgglomerativeClustering(2, compute_distances=True)
clustering = model2.fit(df)
data['Cluster2']=clustering.labels_
fig = go.Figure(data=[go.Scatter(x=df['CarCount'],y=df['BikeCount'], mode='markers', marker_color=data['Cluster2'], marker_size = 4)])
fig.show()


# In[8]:


from sklearn.cluster import DBSCAN

model3 = DBSCAN(eps=15, min_samples=5).fit(df)
data['Cluster3'] = model3.labels_
fig = go.Figure(data=[go.Scatter(x=df['CarCount'],y=df['BikeCount'], mode='markers', marker_color=data['Cluster3'], marker_size = 4)])
fig.show()





