
# coding: utf-8

# # Lab4.House Price Prediction uding LR with Regularization

# #### Step1.Import dataset

# In[180]:


import pandas as pd


# In[181]:


data=pd.read_csv("Ames_House_Sales_Cropped.csv")
data


# In[182]:


#head
data.head()


# In[183]:


#shape
data.shape


# In[184]:


#columns
data.shape[1]


# In[185]:


#dtype
data.dtypes


# In[186]:


#info
data.info()


# In[187]:


#value_counts
data.CentralAir.value_counts()


# #### Step2.Predict Sale Price without Categorical features

# In[188]:


df=data.drop("BldgType",axis=1)
print(df)


# In[189]:


df.drop("CentralAir",axis=1)


# In[190]:


df3=data.pop("BldgType")
print(df3)


# In[191]:


df=data.pop("CentralAir")
df


# In[192]:


data


# In[193]:


X=df1.drop("SalePrice",axis=1)
print(X)


# In[194]:


y=data[["SalePrice"]]
y


# In[195]:


from sklearn.model_selection import train_test_split


# In[196]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[197]:


X_train


# In[198]:


X_test


# In[199]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[200]:


y_pred=lr.predict(X_test)
y_pred


# In[201]:


from sklearn.metrics import mean_squared_error


# In[202]:


mse_In=mean_squared_error(y_test,y_pred)
mse_In


# #### Step3.Create Scatter Plot

# In[203]:


import matplotlib.pyplot as plt


# In[204]:


plt.scatter(y_test,y_pred)


# #### Step4.Encode Categorical columns

# In[205]:


gd=pd.get_dummies(data)
gd


# #### Step5.Predict Sale Price with Categorical features

# In[206]:


x=gd.drop("SalePrice",axis=1)
x


# In[207]:


Y=gd.pop('SalePrice')
Y


# In[208]:


from sklearn.model_selection import train_test_split


# In[209]:


X_train,X_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=42)


# In[210]:


X_train


# In[211]:


from sklearn.linear_model import LinearRegression
lrr=LinearRegression()
lrr.fit(X_train,y_train)


# In[212]:


y_pred=lrr.predict(X_test)
y_pred


# In[213]:


from sklearn.metrics import mean_squared_error


# In[214]:


MSE=mean_squared_error(y_test,y_pred)
MSE


# #### Step6.Normalize using StandardScaler and Predict Sale Price

# In[215]:


from sklearn.preprocessing import StandardScaler


# In[216]:


scaler=StandardScaler()


# In[217]:


scaled_X_train=scaler.fit_transform(X_traStandardScalerin)
scaled_X_train


# In[218]:


scaled_X_train.shape


# In[219]:


scaled_X_test=scaler.transform(X_test)
scaled_X_test


# In[220]:


from sklearn.metrics import mean_squared_error


# In[221]:


Scaler=StandardScaler()
scaled_X_train=Scaler.fit_transform(X_train)
scaled_X_train


# In[222]:


scaled_X_test=Scaler.transform(X_test)
scaled_X_test


# In[223]:


model1=LinearRegression()
model1.fit(scaled_X_train,y_train)
sy_pred=model1.predict(scaled_X_test)
sy_pred


# In[224]:


mse_In1=mean_squared_error(y_test,sy_pred)
mse_In1
print("mean square error using standard scalar:",mse_In1)


# #### Step7.Normalize using MinMaxScaler and Predict Sale Price

# In[225]:


from sklearn.preprocessing import MinMaxScaler
mm_scaler=MinMaxScaler()


# In[226]:


mmX_train=mm_scaler.fit_transform(X_train)
mmX_train


# In[227]:


mmX_test=mm_scaler.transform(X_test)
mmX_test


# In[228]:


model3=LinearRegression()
model3.fit(mmX_train,y_train)


# In[229]:


mms_y_pred=model3.predict(mmX_test)
mms_y_pred


# In[230]:


mmMSE=mean_squared_error(y_test,mms_y_pred)


# In[231]:


print("Mean sqaure error using MinMaxScaler:",mmMSE)


# #### Step8.Predict using Ridge and SGD Regression

# In[239]:


from sklearn.linear_model import SGDRegressor
sgd=SGDRegressor()
sgd.fit(scaled_X_train, y_train)
sgd_y_pred=sgd.predict(scaled_X_test)
print("Predictions of scaled data using SGDRegressor:", sgd_y_pred)


# In[240]:


sgd_mse=mean_squared_error(y_test, sgd_y_pred)
print("SGD_MSE:",sgd_mse)


# In[241]:


from sklearn.linear_model import Ridge
ridge=Ridge()
ridge.fit(scaled_X_train, y_train)
ridge_y_pred=ridge.predict(scaled_X_test)
print("Predictions of scaled data using RIDGERegression:", ridge_y_pred)


# In[242]:


ridge_mse=mean_squared_error(y_test, ridge_y_pred)
print("RIDGE_MSE:",ridge_mse)


# In[ ]:


from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(scaled_X_train, y_train)
lasso_y_pred=lasso.predict(scaled_X_test)
print("Predictions of scaled data using LASSORegression:", lasso_y_pred)


# In[243]:


lasso_mse=mean_squared_error(y_test, lasso_y_pred)
print("LASSO_MSE:",lasso_mse)


# #### Step9.RMSE

# In[246]:


import numpy as np
#RMSE without CD
print("RMSE without one hot encoding: ",np.sqrt(mse_In))
#RMSE with CD
print("RMSE with one hot encoding: ",np.sqrt(MSE))
#RMSE with CD and Standard Scaling
print("RMSE with OHE and SS: ",np.sqrt(mse_In1))
#RMSE with CD and MinMaxScaling
print("RMSE with OHE and MinMaxScaling: ",np.sqrt(mmMSE))
#RMSE of SGDRegressor with CD and StandardScaler
print("RMSE of SGDRegressor with OHE and StandardScaler: ",np.sqrt(sgd_mse))
#RMSE of Ridgecv with CD and Standard Scaler
print("RMSE of Ridgecv with OHE and Standard Scaler: ",np.sqrt(ridge_mse))
#RMSE of LassoCV with CD and StandardScaler
print("RMSE of LassoCV with OHE and StandardScaler",np.sqrt(lasso_mse))

