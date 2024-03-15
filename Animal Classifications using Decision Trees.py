
# coding: utf-8

# # Lab-8. Animal Classifications using Decision Trees

# ### Step1.[Create dataset]

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('data.csv')


# In[3]:


df


# In[4]:


#head
df.head()


# In[5]:


#shape
df.shape


# In[6]:


#size
df.size


# In[7]:


#desceibe
df.describe()


# ### Step2.[Model building using ID3]

# In[8]:


import warnings
warnings.filterwarnings('ignore')


# In[9]:


#create DT model using 'entropy' criterion
X = df.drop(['species'],axis = 1)
y = df['species']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[10]:


from sklearn.tree import DecisionTreeClassifier
clf_entropy = DecisionTreeClassifier(criterion = "entropy")


# In[11]:


#perform training and testing
clf_entropy =clf_entropy.fit(X_train,y_train)
y_pred_entropy = clf_entropy.predict(X_test)
y_pred_entropy


# In[12]:


#accuracy
from sklearn.metrics import accuracy_score
print ("Accuracy for ID3: ",accuracy_score(y_test,y_pred_entropy))


# In[13]:


#classification
from sklearn.metrics import classification_report
print("Classification Report of ID3 : ",classification_report(y_test, y_pred_entropy))


# In[14]:


#interpreting results
from sklearn import tree


# In[15]:


#Visualilze your DT model using graphviz
with open('tree1.dot','w') as f:
 f = tree.export_graphviz(clf_entropy,
 out_file=f,
max_depth=4,
impurity=False,
feature_names = X.columns.values,
 class_names = ['Reptile','Mammal'],
 filled=True)


# In[16]:


get_ipython().system('type tree1.dot')


# In[17]:


tree.plot_tree(clf_entropy)


# ### Step3. [Create a Test Set]

# In[19]:


test_file = pd.read_csv('test_file.csv')


# In[20]:


test_file


# ### Step4. [Perform prediction]

# In[21]:


y_pred_entropy=clf_entropy.predict(test_file)
y_pred_entropy


# ### Step5. [Build DT with zoo dataset]

# In[22]:


clf_gini = DecisionTreeClassifier(criterion = "gini")


# In[23]:


#Train model with full training data
clf_gini.fit(X,y)


# In[24]:


#Predict Samples for the test file
y_pred_gini = clf_gini.predict(test_file)
y_pred_gini


# In[25]:


#Visualize your CART DT using graphviz
with open("tree2.dot",'w') as f:
 f= tree.export_graphviz(clf_gini, out_file=f, max_depth=4, impurity= False, 
 feature_names = X.columns.values, class_names = ['Reptile','Mammal'],filled=True)


# In[26]:


get_ipython().system('type tree2.dot')


# In[27]:


tree.plot_tree(clf_gini)


# ### Step-6. [Build DT with Zoo dataset]

# In[28]:


zoo_df = pd.read_csv('zoo.data')


# In[29]:


zoo_df


# In[30]:


#Split Zoo data into train and test sets
X_zoo = zoo_df.drop(['aardvark', '1.7'], axis = 1)
y_zoo = zoo_df['1.7']


# In[31]:


X_train_zoo,X_test_zoo,y_train_zoo,y_test_zoo = train_test_split(X_zoo, y_zoo, test_size=0.33, random_state=0)


# In[32]:


#Create DT model using 'entropy' criterion
clf_entropy_zoo = DecisionTreeClassifier( criterion = "entropy")
clf_entropy_zoo.fit(X_train_zoo, y_train_zoo)


# In[33]:


y_pred_entropy_zoo = clf_entropy_zoo.predict(X_test_zoo)


# In[34]:


y_pred_entropy_zoo


# In[35]:


#predict using train zoo
y_pred_train_zoo = clf_entropy_zoo.predict(X_train_zoo)
y_pred_train_zoo


# #### Accuracy

# In[36]:


print("Train Accuracy of Zoo Data :", accuracy_score(y_train_zoo, y_pred_train_zoo))


# In[37]:


print("Test Accuracy of Zoo Data :", accuracy_score(y_test_zoo, y_pred_entropy_zoo))


# In[38]:


#Report for zoo data using entropy criterion
print("Report for zoo data using entropy criterion :\n", classification_report(y_test_zoo, y_pred_entropy_zoo))


# In[39]:


#Visualize ID3 DT using graphviz for zoo data
with open("tree1_zoo.dot",'w') as f:
 f= tree.export_graphviz(clf_entropy, out_file=f, max_depth=4, impurity= False, 
 feature_names = X.columns.values,class_names = ['Reptile','Mammal'],filled=True)


# In[40]:


get_ipython().system('type tree1_zoo.dot')


# In[41]:


tree.plot_tree(clf_entropy_zoo)


# ### [Create DT model using 'gini' criterion]

# In[42]:


clf_gini_zoo = DecisionTreeClassifier( criterion = "gini")
clf_gini_zoo.fit(X_train_zoo, y_train_zoo)
y_pred_gini_zoo = clf_gini_zoo.predict(X_test_zoo)


# In[43]:


with open("tree2_zoo.dot",'w') as f:
 f= tree.export_graphviz(clf_gini, out_file=f, max_depth=4, impurity= False, 
 feature_names = X.columns.values,class_names = ['Reptile','Mammal'],filled=True)


# In[44]:


get_ipython().system('type tree2_zoo.dot')


# In[45]:


tree.plot_tree(clf_gini_zoo)

