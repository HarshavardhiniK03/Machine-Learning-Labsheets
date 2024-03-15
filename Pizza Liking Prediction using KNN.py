
# coding: utf-8

# # Pizza Liking Prediction using KNN

# In[4]:


import pandas as pd


# #### Step2.Import dataset

# In[10]:


pizza_data=pd.read_csv('pizza.csv')
pizza_data.head()


# In[11]:


#shape
pizza_data.shape


# In[12]:


#column
pizza_data.shape[1]


# In[30]:


#info
pizza_data.info()


# #### Step3.Visualize Relationships

# In[43]:


pip install seaborn==0.9.0


# In[126]:



df=pd.read_csv('pizza.csv')
df.plot(kind='line',x='age',y='weight')


# #### Step4.Prepare X matrix and Y vector

# In[49]:


x=pd.DataFrame(pizza_data)
cols=[0,1]
x=x[x.columns[cols]]


# In[50]:


y=pizza_data['likepizza'].values


# #### Step5.Examine X and Y 

# In[51]:


x


# In[52]:


type(x)


# In[53]:


y


# In[54]:


type(y)


# #### Step6.Model building

# In[57]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x,y)


# #### Step7.Model testing

# In[59]:


knn.predict(x)


# In[69]:


a=[10,30]
knn.predict([a])


# In[70]:


b=[60,60]
knn.predict([b])


# #### Step8.change n_neighbours=3

# In[73]:


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x,y)


# In[75]:


c=[25,50]
knn.predict([c])


# In[76]:


d=[60,60]
knn.predict([d])


# #### Step9.Predict on entire dataset

# In[77]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x,y)


# In[78]:


y_pred=knn.predict(x)
y_pred


# #### Step10.Accuracy function

# In[79]:


def accuracy(actual,pred):
    return sum(actual==pred)/float(actual.shape[0])


# #### Step11.Find accuracy

# In[80]:


accuracy_score=accuracy(y,y_pred)
accuracy_score


# #### Step12.Prediction on Test Set

# In[82]:


import pandas as pd


# In[87]:


df=pd.read_csv('pizza_test.csv')
df.head()


# In[92]:


#shape
df.shape


# In[95]:


#column
df.shape[1]


# In[97]:


df.info()


# In[98]:


x=pd.DataFrame(df)
cols=[0,1]
x=x[x.columns[cols]]


# In[99]:


x


# In[100]:


Y=df['likepizza'].values
Y


# In[103]:


from sklearn.neighbors import KNeighborsClassifier
test=KNeighborsClassifier(n_neighbors=2)
test.fit(x,Y)


# In[104]:


Y_pred=test.predict(x)
Y_pred


# In[105]:


import numpy as np
Y=np.array([1,1,0,0])
Y


# In[106]:


Y_test=accuracy(Y,Y_pred)
Y_test


# #### Step13.Find best value for K

# In[113]:


scores=[]
for k in range(1,4):
    kn=KNeighborsClassifier(n_neighbors=k)
    kn.fit(x,Y)
    kn.predict(x)
    y_test=kn.predict(x)
    a=accuracy(Y,Y_pred)
    scores.append((k,a))
print(scores)


# #### Step14.accuracy_score function

# In[114]:


from sklearn.metrics import accuracy_score


# In[115]:


accuracy_score(Y,Y_pred)

