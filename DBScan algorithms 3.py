#!/usr/bin/env python
# coding: utf-8

# # DBScan3

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


# In[2]:


# read csv files
df = pd.read_csv(r"C:\Users\DEEPAK KORI\Desktop\ML Assigenment\ML2\cluster_circles.csv")


# In[3]:


df.head()


# In[4]:


#null value 
df.isnull().sum()


# In[5]:


# shape of dataset
df.shape


# In[6]:


df= df.iloc[:,[0,1]].values


# In[7]:


df


# In[8]:


df


# In[9]:


plt.scatter(df[:,0], df[:,1],s=3, c= "blue")


# In[10]:


dbscan = DBSCAN(eps=0.09, min_samples =3)


# In[11]:


lables = dbscan.fit_predict(df)


# In[12]:


np.unique(lables)


# In[13]:


#scaling data
scaling=StandardScaler()
scaled=scaling.fit_transform(df)


# In[14]:


scaled_df=pd.DataFrame(scaled)

# princt scaled dataset
scaled_df.head()


# In[15]:


#chosing no. of clusters as 3 and refitting kmeans model
kmeans = KMeans(n_clusters = 3,random_state = 1)
kmeans.fit(scaled_df)


# In[16]:


#predicting values
cluster_labels = kmeans.fit_predict(scaled_df)


# In[17]:


#labelings
preds = kmeans.labels_
kmeans_df = pd.DataFrame(df)
kmeans_df['KMeans_Clusters'] = preds
# kmeans_df.head()


# # Visualization of clusters for DBSCAN and K-means clustering

# In[18]:


#visulization of clusters in Kmeans
sns.scatterplot(kmeans_df[0],kmeans_df[1],data=kmeans_df ,hue='KMeans_Clusters') 
plt.title("circle", fontsize=15)
plt.xlabel("X1", fontsize=10)
plt.ylabel("X2", fontsize=12)
plt.show()

plt.scatter(df[lables==-1,0],df[lables ==-1,1],s=3 ,c ="blue")
plt.scatter(df[lables==0,0],df[lables ==0,1],s=3 ,c ="red")
plt.scatter(df[lables==1,0],df[lables ==1,1],s=3 ,c ="black")
# plt.scatter(df[lables==2,0],df[lables ==2,1],s=3 ,c ="green")

plt.show()


# # Silhouette score comparison

# In[19]:


#calculate how good our model is
#calculate Silhouette Coefficient for K=3
print(metrics.silhouette_score(df, dbscan.labels_))
print(metrics.silhouette_score(df, kmeans.labels_))


# In[ ]:




