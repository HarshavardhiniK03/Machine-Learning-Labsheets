
# coding: utf-8

# # Lab5: Diabetes Classification using Logistic Regression

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv('diabetes.csv')


# ### Step1. [Understand Data]

# In[5]:


#head
df.head()


# In[7]:


#shape
df.shape


# In[8]:


#columns
df.columns


# In[9]:


#dtype
df.dtypes


# In[10]:


#info
df.info


# In[13]:


#value_counts
df.Glucose.value_counts


# ### Step2. [Build Logistic Regression Model]

# In[44]:


X=df.drop('Outcome',axis=1)


# In[45]:


X.head()


# In[46]:


y=df['Outcome'].values


# In[47]:


y


# In[48]:


from sklearn.model_selection import StratifiedShuffleSplit
s=StratifiedShuffleSplit(n_splits=4,test_size=0.25,random_state=42)


# In[49]:


s.get_n_splits(X,y)


# In[50]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=.25,random_state=42)


# In[52]:


from sklearn.linear_model import LogisticRegression
LOR=LogisticRegression(penalty='l2',C=10.0)
LOR=LOR.fit(X_train,y_train)


# In[53]:


y_pred=LOR.predict(X_test)
y_pred


# ### Step-3. [Predict on a new sample]

# #### Accuracy

# In[54]:


def accuracy(actual,pred):
 return sum(actual==pred)/float(actual.shape[0])


# In[55]:


accuracy_score=accuracy(y_test,y_pred)
accuracy_score


# #### Precision

# In[56]:


from sklearn.metrics import precision_score
print(precision_score(y_test,y_pred))


# #### Recall

# In[57]:


from sklearn.metrics import recall_score
print(recall_score(y_test,y_pred))


# #### AUC Scores

# In[111]:


from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test,y_pred))


# ### step-4. [Understand Correlation]

# In[59]:


from sklearn.metrics import confusion_matrix
cfm=confusion_matrix(y_test,y_pred)
cfm


# In[61]:


import seaborn as sns
sns.heatmap(cfm, annot=True)


# ### Step-5. [Normalization using MinMaxScaler and rebuild LOR]

# In[63]:


#Normalizing using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
mm_X_train=mm.fit_transform(X_train)
mm_X_train


# In[64]:


mm_X_test=mm.transform(X_test)
mm_X_test


# In[65]:


#ReBuild LOR Model
mm_lor=LogisticRegression()
mm_lor=mm_lor.fit(mm_X_train,y_train)


# In[66]:


mm_y_pred=mm_lor.predict(mm_X_test)
mm_y_pred


# #### Accuracy

# In[67]:


def accuracy(actual,pred):
 return sum(actual==pred)/float(actual.shape[0])


# In[68]:


accuracy_score=accuracy(y_test,mm_y_pred)
accuracy_score


# #### Precision

# In[69]:


print(precision_score(y_test,mm_y_pred))


# #### Recall

# In[70]:


print(recall_score(y_test,mm_y_pred))


# #### AUC scores

# In[71]:


mm_auc=print(roc_auc_score(y_test,mm_y_pred))
mm_auc


# ### Step-6: [Normalization using StandardScaler and rebuild LOR]

# In[72]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_X_train=ss.fit_transform(X_train)
ss_X_train


# In[73]:


ss_X_test=ss.transform(X_test)
ss_X_test


# In[74]:


#Rebuild LOR
ss_lor=LogisticRegression()
ss_lor.fit(ss_X_train,y_train)
ss_y_pred=ss_lor.predict(ss_X_test)


# #### Accuracy

# In[75]:


def accuracy(actual,pred):
 return sum(actual==pred)/float(actual.shape[0])


# In[76]:


ss_accuracy_score=accuracy(y_test,ss_y_pred)
ss_accuracy_score


# #### Precision

# In[77]:


print(precision_score(y_test,ss_y_pred))


# #### Recall

# In[78]:


print(recall_score(y_test,ss_y_pred))


# #### AUC scores

# In[79]:


auc_ss=print(roc_auc_score(y_test,ss_y_pred))
auc_ss


# ### Step-7. [Plot ROC Curve]

# In[83]:


pred_prob1=mm_lor.predict_proba(mm_X_test)


# In[84]:


from sklearn.metrics import roc_curve
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)


# In[85]:


import matplotlib.pyplot as plt
plt.plot(fpr1,tpr1,linestyle='--',color='orange',label='MinMaxScaler values')


# ### Step-8. [Comparison with KNN classifier]

# In[86]:


from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=4)
KNN=KNN.fit(X_train,y_train)


# In[87]:


kNN_y_pred=knn.predict(X_test)


# In[88]:


from sklearn.preprocessing import MinMaxScaler
m=MinMaxScaler()
m_X_train=m.fit_transform(X_train)
m_X_train


# In[89]:


m_X_test=m.transform(X_test)
m_X_test


# In[90]:


m_KNN=KNeighborsClassifier()
m_KNN=m_KNN.fit(m_X_train,y_train)
m_y_pred=m_KNN.predict(m_X_test)
m_y_pred


# ### Classification Metrics

# #### Accuracy

# In[92]:


def accuracy(actual,pred):
 return sum(actual==pred)/float(actual.shape[0])


# In[93]:


m_accuracy_score=accuracy(y_test,m_y_pred)
m_accuracy_score


# ### Precision

# In[ ]:


print(precision_score(y_test,m_y_pred))


# ### Recall

# In[94]:


print(recall_score(y_test,m_y_pred))


# #### AUC Scores

# In[95]:


KNN_auc=print(roc_auc_score(y_test,m_y_pred))
KNN_auc


# ### Step-9. [Update ROC Curve]

# In[96]:


pred_prob2=m_KNN.predict_proba(m_X_test)


# In[97]:


from sklearn.metrics import roc_curve
fpr2,tpr2,thresh2=roc_curve(y_test,pred_prob2[:,1],pos_label=1)


# In[98]:


import matplotlib.pyplot as plt
plt.plot(fpr1,tpr1,linestyle='--',color='orange',label='MinMaxScaler values')
plt.plot(fpr2,tpr2,linestyle='--',color='brown',label='KNN Classifier')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show()


# ### Step-10. [Regularization]

# In[105]:


from sklearn.linear_model import LogisticRegressionCV
model1=LogisticRegressionCV(Cs=10,cv=4,penalty='l1',solver='liblinear')
model2=LogisticRegressionCV(Cs=10,cv=4,penalty='l2')
model1.fit(mm_X_train,y_train)
model2.fit(mm_X_train,y_train)


# In[107]:


rg_y_pred1 = model1.predict(mm_X_test)
rg_y_pred2 = model2.predict(mm_X_test)


# #### AUC SCORE OF L1

# In[108]:


from sklearn.metrics import roc_auc_score
l1_auc = roc_auc_score(y_test, rg_y_pred1)
l1_auc = (' LOR L1 MINMAX AUC', l1_auc)
l1_auc


# #### AUC SCORE OF L2

# In[109]:


from sklearn.metrics import roc_auc_score
l2_auc = roc_auc_score(y_test, rg_y_pred2)
l2_auc = (' LOR L2 MINMAX AUC', l2_auc)
l2_auc


# ### STEP 11 : UPDATE ROC CURVE

# In[110]:


pred_prb7 = model1.predict_proba(mm_X_test)
pred_prb8 = model2.predict_proba(mm_X_test)
fpr,tbr,threshold = roc_curve(y_test, pred_prob1[:,1],pos_label=1)
fpr1,tbr1,threshold1 = roc_curve(y_test, pred_prob2[:,1],pos_label=1)
fpr2,tbr2,threshold2= roc_curve(y_test, pred_prb7[:,1],pos_label=1)
fpr3,tbr3,threshold3 = roc_curve(y_test, pred_prb8[:,1],pos_label=1)
plt.plot(fpr, tbr, linestyle='-', color='brown', label='LogisticRegression')
plt.plot(fpr1, tbr1, linestyle='-', color='red', label='KNN')
plt.plot(fpr3, tbr3, linestyle='-', color='midnightblue', label='l2')
plt.plot(fpr2, tbr2, linestyle='-', color='black', label='l1')
plt.annotate(xy=[0.5,0.3],s= auc_ss)
plt.annotate(xy=[0.5,0.2],s= knn_auc)
plt.annotate(xy=[0.5,0.1],s= l1_auc)
plt.annotate(xy=[0.7,0],s= l2_auc)
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'best')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

