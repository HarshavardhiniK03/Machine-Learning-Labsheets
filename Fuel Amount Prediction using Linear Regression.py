
# coding: utf-8

# ## Lab3. Fuel Amount Prediction using Linear Regression

# In[1]:


import pandas as pd


# ### Step2.Import dataset

# In[3]:


df=pd.read_csv("fuel_data.csv")
df


# In[4]:


#head
df.head()


# In[5]:


#shape
df.shape


# In[6]:


#columns
df.columns


# In[7]:


#type
type(df)


# In[8]:


#info
df.info()


# ### Step3.Preprocessing

# In[97]:


df.isnull()


# ### Step.4 Visualize Relationships.

# In[ ]:


import numpy as np
import seaborn as sns


# In[ ]:


sns.relplot(data=df,x='drivenKM',y='fuelAmount')


# ### Step5.Prepare X matrix and y matrix

# In[48]:


x=df[['drivenKM']]
y=df[['fuelAmount']]


# In[49]:


x


# In[50]:


y


# ### Step6.Examine X and y.

# In[51]:


type(x)


# In[52]:


type(y)


# ### Step7.Split dataset

# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[55]:


x_train


# In[56]:


x_train.shape


# In[57]:


y_train


# In[58]:


y_train.shape


# In[59]:


x_test


#  x_test.shape

# In[60]:


y_test


# In[61]:


y_test.shape


# ## Part-I Linear Regression Baseline Model

# ### Step8.Build Model

# In[62]:


from sklearn.linear_model import LinearRegression
im=LinearRegression()
im.fit(x_train,y_train)


# ### Step9.Predict price for 800 KM 

# In[63]:



price=[[800]]
im.predict(price)


# ### Step10.Predict on entire dataset.

# In[64]:


y_data=im.predict(x_test)
y_data


# ### Step11.Print Mean Squared Error and R2 Error

# In[65]:


from sklearn.metrics import mean_squared_error


# In[66]:


mean_squared_error(y_test,y_data)


# In[67]:


im.coef_


# In[68]:


im.intercept_


# ### Part - II Linear Regression With Scaling Using StandardScaler

# ### Step12.Normalize X_train and X_test values.

# In[69]:


from sklearn.preprocessing import StandardScaler


# In[70]:


scaler=StandardScaler()


# In[71]:


norm_x_train=scaler.fit_transform(x_train)
norm_x_train


# In[72]:


norm_x_train.shape


# In[73]:


norm_x_test=scaler.transform(x_test)
norm_x_test


# ### Step13.Build LR model

# In[74]:


from sklearn.linear_model import LinearRegression


# In[76]:


LR_model=LinearRegression()
LR_model.fit(scaled_x_train,y_train)
y_predict=LR_model.predict(norm_x_test)
y_predict


# In[77]:


from sklearn.metrics import mean_squared_error


# In[78]:


LR=mean_squared_error(y_test,y_predict)
LR


# ### Step15.Plot scatter plot.

# In[80]:


import matplotlib.pyplot as plt


# In[81]:


plt.scatter(y_test,y_predict)


# ### Part-III.Linear Regression with Scaling using MinMaxScaler and Comparison with KNeighborsRegressor and SGDRegressor.

# ### Step16.Repeat with MinmaxScaler.

# In[82]:


from sklearn.preprocessing import MinMaxScaler


# In[84]:


mms=MinMaxScaler()
mms.fit(scaled_x_train,y_train)
mms_pe=mms.transform(norm_x_test)
mms_pe


# In[85]:


MMS=mean_squared_error(y_test,mms_pe)
MMS


# ### Step17.Compare KNN Regressor

# In[86]:


from sklearn.neighbors import KNeighborsRegressor


# In[88]:


neigh = KNeighborsRegressor()
neigh.fit(scaled_x_train,y_train)
re=neigh.predict(norm_x_test)


# In[89]:


KN=mean_squared_error(y_test,re)
KN


# ### Step18.Compare SGD Regressor.

# In[90]:


from sklearn.linear_model import SGDRegressor


# In[92]:


sgd=SGDRegressor()
sgd.fit(scaled_x_train,y_train)
sgd_pre=sgd.predict(norm_x_test)
sgd_pre


# In[93]:


SGDR1=mean_squared_error(y_test,sgd_pre)
SGDR1


# ### Step19.Select best model.

# In[94]:


table=pd.DataFrame([SGDR1,KN,LR])


# In[95]:


table['Algorithm']=['SGDR','KN',"LR"]


# In[96]:


table.set_index('Algorithm')

