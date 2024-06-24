#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings(action='ignore')


# In[5]:


pd.set_option('display.max_columns',10,'display.width',1000)
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.head()


# In[6]:


train.shape

test.shape
# In[8]:


test.shape


# In[9]:


train.isnull().sum()


# In[10]:


test.isnull().sum()


# In[12]:


train.describe(include="all")


# In[23]:


male_ind = len(train[train['Sex'] == 'male'])
print("No of males in titanic:",male_ind)


# In[25]:


female_ind = len(train[train['Sex'] == 'female'])
print("No of females in titanic:",female_ind)


# In[27]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
gender = ['Male','Female']
index=[577,314]0[P]
ax.bar(gender,index)
plt.xlabel("Gender")
plt.ylabel("No of people onboarding ship")
plt.show()


# In[28]:


alive=len(train[train['Survived']==1])
dead=len(train[train['Survived']==0])


# In[29]:


train.groupby('Sex')[['Survived']].mean()


# In[31]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
status=['Survived','Dead']
ind = [alive,dead]
ax.bar(status,ind)
plt.xlabel("status")
plt.show()


# In[35]:


plt.figure(1)
train.loc[train['Survived']==1,'Pclass'].value_counts().sort_index().plot.bar()
plt.title('Bar graph of people according to ticket class in which people survived')

plt.figure(2)
train.loc[train['Survived']==0,'Pclass'].value_counts().sort_index().plot.bar()
plt.title('Bar graph of people according to ticket class in which people couldn\'t survived')


# In[39]:


plt.figure(1)
age=train.loc[train.Survived==1,'Age']
plt.title("The histogram of the age groups of the people that had survived ")
plt.hist(age, np.arange(0,100,10))
plt.xticks(np.arange(0,100,10))

plt.figure(2)
age=train.loc[train.Survived==0,'Age']
plt.title("The histogram of the age groups of the people that couldn\'t survived ")
plt.hist(age, np.arange(0,100,10))
plt.xticks(np.arange(0,100,10))



# In[40]:


train[["SibSp","Survived"]].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[41]:


train[["Age","Survived"]].groupby(['Age'],as_index=False).mean().sort_values(by='Survived',ascending=True)


# In[42]:


train[["Embarked","Survived"]].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[45]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l=['C=Cherbourg','Q=Queenstown','S=Southampton']
s=[0.553571,0.389610,0.336957]
ax.pie(s,labels=l,autopct='%1.2f%%')
plt.show()


# In[46]:


test.describe(include="all")


# In[52]:


train = train.drop(['Cabin'],axis = 1)
test = test.drop(['Cabin'],axis = 1)


# In[54]:


train = train.drop(['Name'],axis = 1)
test = test.drop(['Name'],axis = 1)


# In[57]:


column_train=['Age','Pclass','SibSp','Parch','Fare','Sex','Embarked']
x=train[column_train]
y=train['Survived']


# In[58]:


x['Age'].isnull().sum()
x['Pclass'].isnull().sum()
x['SibSp'].isnull().sum()
x['Parch'].isnull().sum()
x['Fare'].isnull().sum()
x['Sex'].isnull().sum()
x['Embarked'].isnull().sum()


# In[59]:


x['Age']=x['Age'].fillna(x['Age'].median())
x['Age'].isnull().sum()


# In[60]:


x['Pclass']=x['Pclass'].fillna(x['Pclass'].median())
x['Pclass'].isnull().sum()


# In[61]:


x['SibSp']=x['SibSp'].fillna(x['SibSp'].median())
x['SibSp'].isnull().sum()


# In[62]:


x['Parch']=x['Parch'].fillna(x['Parch'].median())
x['Parch'].isnull().sum()


# In[63]:


x['Fare']=x['Fare'].fillna(x['Fare'].median())
x['Fare'].isnull().sum()


# In[66]:


x['Embarked']=train['Embarked'].fillna(method='pad')
x['Embarked'].isnull().sum()


# In[67]:


d={'male':0,'female':1}
x['Sex']=x['Sex'].apply(lambda x:d[x])


# In[68]:


x['Sex'].head()


# In[73]:


e={'C':0,'Q':1,'S':2}
x['Embarked']=x['Embarked'].apply(lambda x:e[x])
x['Embarked'].head()


# In[79]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=7)


# In[80]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
print("Acuuracy Score:",accuracy_score(y_test,y_pred))


# In[82]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_mat = confusion_matrix(y_test,y_pred)
print(confusion_mat)


# In[86]:


from sklearn.svm import SVC
model1=SVC()
model1.fit(x_train,y_train)

pred_y=model1.predict(x_test)

print("Acc:",accuracy_score(y_test,y_pred))


# In[87]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test,y_pred)
print(confusion_mat)
print(classification_report(y_test,y_pred))


# In[89]:


from sklearn.neighbors import KNeighborsClassifier
model2=KNeighborsClassifier(n_neighbors=5)
model2.fit(x_train,y_train)
y_pred2=model2.predict(x_test)


# In[90]:


print("Acc:",accuracy_score(y_test,y_pred2))


# In[91]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test,y_pred2)
print(confusion_mat)
print(classification_report(y_test,y_pred2))


# In[93]:


from sklearn.naive_bayes import GaussianNB
model3=GaussianNB()
model3.fit(x_train,y_train)
y_pred3=model3.predict(x_test)

print("Acc:",accuracy_score(y_test,y_pred3))


# In[94]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test,y_pred3)
print(confusion_mat)
print(classification_report(y_test,y_pred3))


# In[95]:


from sklearn.tree import DecisionTreeClassifier
model4=DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(x_train,y_train)
y_pred4=model4.predict(x_test)

print("Acc:",accuracy_score(y_test,y_pred4))


# In[96]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test,y_pred4)
print(confusion_mat)
print(classification_report(y_test,y_pred4))


# In[98]:


results=pd.DataFrame({
    'Model':['Logistic regression','Support Vector Machines','Naive Bayes','Knn','Decision Tree'],
    'Score':[0.7611,0.76119,0.7686,0.6604,0.7425]
})
results_df=results.sort_values(by='Score',ascending=True)
results_df=results_df.set_index('Score')
results_df.head()


# In[ ]:




