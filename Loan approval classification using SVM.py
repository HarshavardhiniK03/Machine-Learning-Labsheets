
# coding: utf-8

# ### STEP 1:

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv("train_loan.csv")
df


# In[4]:


df.head()


# In[5]:


df.shape


# In[26]:


df.columns


# In[27]:


df.dtypes


# In[28]:


df.info


# In[29]:


df.info()


# In[30]:


df.Gender.value_counts


# ### STEP 2:

# In[6]:


df["Dependents"].fillna("NO_dep",inplace=True)


# In[7]:


df["Dependents"]


# In[24]:


dept={"0":0,"1":1,"2":2,"3+":3,"NO_dep":0}
df.Dependents=[dept[item]for item in df.Dependents]


# In[25]:


df['Dependents'].astype(int)


# In[8]:


df['Gender'].fillna(df["Gender"].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Education'].fillna(df['Education'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)      


# In[9]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)


# In[10]:


df.drop(["Loan_ID"],axis=1)


# ### STEP 3:

# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[23]:


sns.countplot(x='Married',data=df)  
plt.show()


# In[14]:


sns.countplot(x='Dependents',data=df)  
plt.show()    


# In[22]:


sns.countplot(x='Education',data=df)  
plt.show()


# In[21]:


sns.countplot(x='Self_Employed',data=df)  
plt.show()


# ### STEP 4:

# In[16]:


x=df.drop(['Loan_Status'],axis=1)


# In[17]:


y=df.pop('Loan_Status')


# ### STEP 5:

# In[19]:


x=pd.get_dummies(x)


# ### STEP 6:

# In[20]:


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)


# In[34]:


from sklearn.preprocessing import StandardScaler 
ss=StandardScaler()


# In[35]:


x_train_ss=ss.fit_transform(x_train)
x_train_ss


# In[36]:


x_test_ss=ss.fit_transform(x_test)
x_test_ss


# In[37]:


from sklearn.svm import LinearSVC
lvc=LinearSVC()
lvc.fit(x_train_ss,y_train)
l_pred=lvc.predict(x_test_ss)
l_pred


# In[63]:


from sklearn.metrics import accuracy_score as acs
lvc_acc=acs(y_test,l_pred)
lvc_acc


# In[50]:


from sklearn.metrics import confusion_matrix as cm
mat=cm(y_test,l_pred)
mat


# In[52]:


from sklearn.metrics import classification_report as cr
cre=cr(y_test,l_pred)
print(cre)


# ### STEP 7:

# In[62]:


from sklearn.linear_model import LogisticRegression 
lor=LogisticRegression()
lor.fit(x_train_ss,y_train)
lr_pred=lor.predict(x_test_ss)

from sklearn.svm import LinearSVC
lvc=LinearSVC()
lvc.fit(x_train_ss,y_train)
l_pred=lvc.predict(x_test_ss)

from sklearn.metrics import accuracy_score as acs
lvc_acc=acs(y_test,l_pred)
print("linear accuracy score:",lvc_acc)

lvc_acc=acs(y_test,lr_pred)
print("logistic regression accuracy score:",lvc_acc)

