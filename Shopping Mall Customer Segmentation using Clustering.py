
# coding: utf-8

# # PML Lab11. Shopping Mall Customer Segmentation using Clustering

# ### STEP -1 UNDERSTAND DATA

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('Mall_Customers.csv')


# In[3]:


# head
df.head()


# In[4]:


#shape
df.shape


# In[5]:


#size
df.size


# In[6]:


#columns
df.columns


# In[7]:


#value_counts
df.Genre.value_counts()


# In[8]:


#info
df.info()


# In[9]:


#dtypes
df.dtypes


# ### STEP - 2 LABEL ENCODE GENDER

# In[10]:


# Genre (ie., gender) is a string, so label encode into binary
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Genre']= label_encoder.fit_transform(df['Genre'])
df['Genre'].unique()


# ### STEP - 3 CHECK FOR VARIANCE

# In[11]:


df.describe()


# In[12]:


df.info()


# In[13]:


df.var()


# In[14]:


df.corr()


# ### STEP 4 CHECK SKEWNESS

# In[15]:


df.skew()


# In[16]:


df.sort_values(by =['Genre','Age','Annual Income (k$)','Spending Score (1-100)'])


# ### STEP 5 PAIR PLOT

# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


sns.pairplot(data=df)


# ### STEP - 6 BUILD KMEANS

# In[19]:


from sklearn.cluster import KMeans


# In[20]:


df.drop(['CustomerID'],axis=1, inplace=True)


# In[21]:


KM = KMeans(n_clusters=5)


# In[22]:


KM.fit(df)


# In[23]:


KM.labels_


# In[24]:


print(KM.cluster_centers_)


# ### STEP - 7 SCATTER PLOT

# In[25]:


import warnings
warnings.filterwarnings('ignore')


# In[26]:


sns.scatterplot(df['Annual Income (k$)'], df['Spending Score (1-100)'], hue=KM.labels_)


# ### STEP - 8 CLUSTER ANALYSIS

# In[27]:


kmeans2 = KMeans(n_clusters = 5, init='k-means++')
kmeans2.fit(df)
pred = kmeans2.predict(df)


# In[28]:


frame = pd.DataFrame(df)
frame['cluster'] = pred


# In[29]:


frame.cluster.value_counts()


# In[30]:


frame


# In[31]:


C0 = df[df['cluster'] == 0]
C1 = df[df['cluster'] == 1]
C2 = df[df['cluster'] == 2]
C3 = df[df['cluster'] == 3]
C4 = df[df['cluster'] == 4]


# In[32]:


import statistics as ss
print('Average Age : ',C0['Age'].mean())
print('Average Annual Income : ',C0['Annual Income (k$)'].mean())
print('Deviation of the mean for annual Income : ',ss.stdev(C0['Annual Income (k$)']))
print('No. of Customers ie shape :' ,C0.shape)
print('From those Customers We have',C0.Genre.value_counts()[1],'male and',C0.Genre.value_counts()[1])


# In[33]:


print('Average Age : ',C1['Age'].mean())
print('Average Annual Income : ',C1['Annual Income (k$)'].mean())
print('Deviation of the mean for annual Income : ',ss.stdev(C1['Annual Income (k$)']))
print('No. of Customers ie shape :' ,C1.shape)
print('From those Customers We have',C1.Genre.value_counts()[1],'male and',C1.Genre.value_counts()[1])


# In[34]:


print('Average Age : ',C2['Age'].mean())
print('Average Annual Income : ',C2['Annual Income (k$)'].mean())
print('Deviation of the mean for annual Income : ',ss.stdev(C2['Annual Income (k$)']))
print('No. of Customers ie shape :' ,C2.shape)
print('From those Customers We have',C2.Genre.value_counts()[1],'male and',C2.Genre.value_counts()[1])


# In[35]:


print('Average Age : ',C3['Age'].mean())
print('Average Annual Income : ',C3['Annual Income (k$)'].mean())
print('Deviation of the mean for annual Income : ',ss.stdev(C3['Annual Income (k$)']))
print('No. of Customers ie shape :' ,C3.shape)
print('From those Customers We have',C3.Genre.value_counts()[1],'male and',C3.Genre.value_counts()[1])


# In[36]:


print('Average Age : ',C4['Age'].mean())
print('Average Annual Income : ',C4['Annual Income (k$)'].mean())
print('Deviation of the mean for annual Income : ',ss.stdev(C4['Annual Income (k$)']))
print('No. of Customers ie shape :' ,C4.shape)
print('From those Customers We have',C4.Genre.value_counts()[1],'male and',C4.Genre.value_counts()[1])


# ### STEP 9 FIND THE BEST NUMBER

# In[37]:


SSE = []
for clust in range(1,20):
    KM = KMeans(n_clusters= clust, init='k-means++')
    KM = KM.fit(df)
    SSE.append(KM.inertia_)


# In[38]:


plt.plot(np.arange(1,20), SSE,'ro-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')


# ### STEP -10 REDUCE DIMESNSION USING PCA

# In[39]:


from sklearn.decomposition import PCA


# In[40]:


pca = PCA(n_components=2)
_PCA = pca.fit_transform(df)
PCA_Components = pd.DataFrame(_PCA)


# In[41]:


PCA_Components


# In[42]:


KM1 = KMeans(n_clusters=5)
KM1.fit(PCA_Components)
KM1.cluster_centers_


# In[43]:


KM1.labels_


# ### STEP 11 SCATTER PLOT

# In[44]:


sns.scatterplot(PCA_Components[0], PCA_Components[1], hue=KM1.labels_)


# ### STEP 12 MEAN SHIFT CLUSTERING

# In[45]:


from sklearn.cluster import MeanShift, AgglomerativeClustering


# In[46]:


MS = MeanShift(bandwidth = 50)
MS.fit(PCA_Components)
MS.cluster_centers_


# In[47]:


sns.scatterplot(PCA_Components[0], PCA_Components[1], hue=KM1.labels_)


# ### STEP 13 PREDICT HIERARCHICAL CLUSTERS USING AGGLOMERATIVE CLUSTERING

# In[48]:


AC = AgglomerativeClustering(n_clusters = 5, linkage='ward',compute_full_tree=True)
AC.fit(df)


# In[49]:


AC.labels_


# In[50]:


df['Cluster'] = AC.labels_


# In[51]:


import scipy.cluster.hierarchy as sch


# In[52]:


from scipy.cluster import hierarchy


# In[53]:


Z = hierarchy.linkage(df[:30], 'ward')
plt.figure(figsize=(10,5))
dn = hierarchy.dendrogram(Z)


# ### STEP 14 VISUALIZE SCATTER PLOT WITH HUE AS AGGLOMERATIVECLUSTERING LABELS_

# In[54]:


sns.scatterplot(df['Annual Income (k$)'], df['Spending Score (1-100)'], hue=AC.labels_)

