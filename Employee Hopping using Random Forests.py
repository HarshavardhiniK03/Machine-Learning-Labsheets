
# coding: utf-8

# # Lab9. Employee Hopping using Random Forests

# ### Step1.[Understand Data]

# In[30]:


import pandas as pd


# In[31]:


df=pd.read_csv('Employee_hopping.csv')
df


# In[32]:


#head
df.head()


# In[33]:


#shape
df.shape


# In[34]:


#columns
df.columns


# In[35]:


#dtype
df.dtypes


# In[36]:


#info
df.info


# In[37]:


#value_counts
df['EducationField'].value_counts


# ### Step2.[Extract X and Y]

# In[38]:


x=df.drop('Attrition',axis=1)
df.Attrition


# In[39]:


y=df['Attrition'].values
y


# ### Step3.[Feature Engineering]

# In[40]:


df = pd.get_dummies(df,columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime'])


# In[41]:


df


# ### Step4: Shape of X and Y

# In[42]:


x.shape


# In[43]:


y.shape


# ### Step5.[Model Development]

# In[46]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state =42)


# In[47]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100, max_features=0.3)


# In[48]:


RFC.fit(X_train,y_train)


# In[49]:


RFC_y_pred = RFC.predict(X_test)
RFC_y_pred


# ### Step-6. [Testing]

# In[50]:


from sklearn.metrics import accuracy_score,classification_report
RFC_acc = accuracy_score(y_test,RFC_y_pred)
RFC_acc


# In[51]:


print(classification_report(y_test, RFC_y_pred))


# ### Step- 7 [FEATURE IMPORTANT VALUE]

# In[52]:


print(RFC.feature_importances_)


# In[54]:


feature_name = pd.DataFrame(RFC.feature_importances_, index=X_train.columns, columns=['Important_Feature'])
feature_name


# In[55]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[62]:


pd.Series(RFC.feature_importances_, index=X_train.columns).sort_values(ascending=False).plot(kind='bar',figsize=(10,10))


# ### Step- 8[Visualize your RF Decision Tree using graphviz]

# In[104]:


estimator = RFC.estimators_[5]
from sklearn import tree
from sklearn.tree import export_graphviz
with open("RFDT.dot", 'w') as f:
    f = tree.export_graphviz(estimator, out_file=f, max_depth=4, impurity=False)


# In[105]:


get_ipython().system('dot- Tpng RFDT.dot -o RFDT.png')


# In[108]:


import matplotlib.pyplot as plt
image = plt.imread('RFDT.png')
plt.figure(figsize=(19,15))
plt.imshow(image)


# ### Step- 9:[RF WITH A RANGE OF TREES]

# In[95]:


import warnings
warnings.filterwarnings('ignore')
rf2 = RandomForestClassifier(oob_score=True, random_state=42, warm_start=True, n_jobs=-1)
oob_list = list()
for n_trees in [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]:
    rf2.set_params(n_estimators=n_trees)
    rf2.fit(X_train, y_train)
    oob_error = 1 - rf2.oob_score_
    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))
rf_oob_df = pd.concat(oob_list, axis=1).T.set_index('n_trees')
rf_oob_df


# ### Step-10 [PLOT OOB -ERROR FOR EACH TREE]

# In[73]:


ax = rf_oob_df.plot(legend=False, marker='o', figsize=(10,5))
ax.set(ylabel='out-of-bag error')


# ### Step- 11 [COMPARE WITH DECISION TREE CLASSIFIER]

# #### >Create DecisionTreeClassifier, fit and predict on test set
# #### >Visualize the tree using graphviz
# #### >Print accuracy score
# #### >Print classification report
# #### >What is the result of the comparision between RF and DT models? Which gives best accuracy?.
# #### >What is your comment on precision, recall, f1 score values?

# In[74]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_test,y_test)


# In[75]:


y_pred1 = clf.predict(X_test)
y_pred1


# In[109]:


from sklearn import tree
from sklearn.tree import export_graphviz
with open("DTC2.dot", 'w') as f:
    f = tree.export_graphviz(clf,out_file=f,max_depth = 4,impurity = False,feature_names=X.columns.values,class_names=['Yes','No'],filled=True)
                         


# In[110]:


get_ipython().system('dot -Tpng DTC2.dot -o DTC2.png')


# In[111]:


image = plt.imread('DTC2.png')
plt.figure(figsize=(19,15))
plt.imshow(image)


# In[112]:


print("Accuracy of test :",clf.score(X_test,y_test))


# In[113]:


print(classification_report(y_test,RFC_y_pred))

