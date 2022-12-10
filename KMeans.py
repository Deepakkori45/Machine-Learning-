#!/usr/bin/env python
# coding: utf-8

# In[12]:


# import libraries

import pandas as pd  #imporitng data from file
import matplotlib.pyplot as plt #data visualization and graphical plot
import seaborn as sns #making statistical graphics
from sklearn import metrics #use for clustering
from sklearn.cluster import KMeans #relationship among data samples.
from sklearn.preprocessing import StandardScaler #standardization of the data set
from sklearn.cluster import AgglomerativeClustering


# In[13]:


# read csv files
dict_df = pd.read_csv(r"C:\Users\DEEPAK KORI\Desktop\data-dictionary.csv")
df = pd.read_csv(r"C:\Users\DEEPAK KORI\Desktop\Country-data.csv")


# In[14]:


# first few rows of dictionary dataset
dict_df.head()


# In[15]:


# first few rows of dictionary dataset
df.head()


# In[16]:


#null value 
df.isnull().sum()


# In[17]:


#drop country column
data=df.drop(['country'],axis=1)


# In[18]:


data.head()


# In[19]:


#correlation 
corr_matrix=data.corr()
sns.heatmap(corr_matrix,annot=True)


# In[20]:


#scaling data
scaling=StandardScaler()
scaled=scaling.fit_transform(data)


# In[24]:


scaled_df=pd.DataFrame(scaled,columns=data.columns)

# princt scaled dataset
scaled_df.head()


# In[25]:


#chosing no. of clusters as 3 and refitting kmeans model
kmeans = KMeans(n_clusters = 3,random_state = 1)
kmeans.fit(scaled_df)


# In[26]:


#calculate how good our model is
#calculate Silhouette Coefficient for K=3

metrics.silhouette_score(scaled_df, kmeans.labels_)


# In[27]:


#predicting values
cluster_labels = kmeans.fit_predict(scaled_df)


# In[28]:


#labelings
preds = kmeans.labels_
kmeans_df = pd.DataFrame(df)
kmeans_df['KMeans_Clusters'] = preds
kmeans_df.head(167)


# In[29]:


#save a kmeans file
kmeans_df.to_csv('kmeans_result.csv',index=False)


# In[30]:


#visulization of clusters inflation vs gdpp
sns.scatterplot(kmeans_df['income'],kmeans_df['gdpp'],hue='KMeans_Clusters',data=kmeans_df) 
plt.title("income vs gdpp", fontsize=15)
plt.xlabel("income", fontsize=12)
plt.ylabel("gdpp", fontsize=12)
plt.show()


# In[31]:


#visulization of clusters child mortality vs gdpp
sns.scatterplot(kmeans_df['child_mort'],kmeans_df['gdpp'],hue='KMeans_Clusters',data=kmeans_df) 
plt.title("Child Mortality vs gdpp", fontsize=15)
plt.xlabel("Child Mortality", fontsize=12)
plt.ylabel("gdpp", fontsize=12)
plt.show()


# In[32]:


#box plot
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
bp=sns.boxplot(y=df.child_mort,ax=ax[0, 0])
ax[0, 0].set_title('Child Mortality Rate')
bp=sns.boxplot(y=df.health,ax=ax[0, 1])
ax[0, 1].set_title('Health')
bp=sns.boxplot(y=df.income,ax=ax[0, 2])
ax[0,2].set_title('Income per Person')
bp=sns.boxplot(y=df.inflation,ax=ax[1, 0])
ax[1,0].set_title('Inflation')
bp=sns.boxplot(y=df.imports,ax=ax[1,1])
ax[1, 1].set_title('Imports')
s=sns.boxplot(y=df.life_expec,ax=ax[1, 2])
ax[1,2].set_title('Life Expectancy')
s=sns.boxplot(y=df.total_fer,ax=ax[2,0])
ax[2,0].set_title('Total Fertility')
s=sns.boxplot(y=df.gdpp,ax=ax[2, 1])
ax[2,1].set_title('GDP per Capita')
s=sns.boxplot(y=df.exports,ax=ax[2,2])
ax[2,2].set_title('Exports')
plt.show()


# In[33]:


#find number of developed country,developing country,under-developed country
under_developing=kmeans_df[kmeans_df['KMeans_Clusters']==0]['country']
developing=kmeans_df[kmeans_df['KMeans_Clusters']==1]['country']
developed=kmeans_df[kmeans_df['KMeans_Clusters']==2]['country']

print("Number of deveoped countries",len(under_developing))
print("Number of developing countries",len(developing))
print("Number of under-developing countries",len(developed))


# In[34]:


#list of developed countries
list(developed)


# In[35]:


#list of developing countries
list(developing)


# In[36]:


for i in developing:
    if i == 'India':
        print('Yes', i , 'is present in developing countries list')  


# In[37]:


#list of under-developing countries
list(under_developing)


# In[38]:


for i in under_developing:
    if i == 'Pakistan':
        print('Yes', i , 'is present in under_developing countries list')


# In[ ]:




