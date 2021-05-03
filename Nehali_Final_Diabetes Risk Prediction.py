#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


df = pd.read_csv('E:/Nehali/MS DS/Data Mining/Final Project/Early-Stage-Diabetes-UCI-ML-master/diabetes_data_upload.csv')


# ## Data Exploration

# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df['Gender'] = df['Gender'].apply(str)


# In[7]:


df['class'].value_counts(), df['Gender'].value_counts()


# In[8]:


df['Age'].plot.hist()
plt.show()


# In[9]:


df.columns


# In[10]:


df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Polyuria'] = df['Polyuria'].map({'Yes': 1, 'No': 0})
df['Polydipsia'] = df['Polydipsia'].map({'Yes': 1, 'No': 0})
df['sudden weight loss'] = df['sudden weight loss'].map({'Yes': 1, 'No': 0})
df['weakness'] = df['weakness'].map({'Yes': 1, 'No': 0})
df['Polyphagia'] = df['Polyphagia'].map({'Yes': 1, 'No': 0})
df['Genital thrush'] = df['Genital thrush'].map({'Yes': 1, 'No': 0})
df['visual blurring'] = df['visual blurring'].map({'Yes': 1, 'No': 0})
df['Itching'] = df['Itching'].map({'Yes': 1, 'No': 0})
df['Irritability'] = df['Irritability'].map({'Yes': 1, 'No': 0})
df['delayed healing'] = df['delayed healing'].map({'Yes': 1, 'No': 0})
df['partial paresis'] = df['partial paresis'].map({'Yes': 1, 'No': 0})
df['muscle stiffness'] = df['muscle stiffness'].map({'Yes': 1, 'No': 0})
df['Alopecia'] = df['Alopecia'].map({'Yes': 1, 'No': 0})
df['Obesity'] = df['Obesity'].map({'Yes': 1, 'No': 0})
df['class'] = df['class'].map({'Positive': 1, 'Negative': 0})


# ## Model Building

# In[11]:


from sklearn.model_selection import train_test_split

X = df.drop(['class'], axis='columns')
y = df['class']


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)


# In[13]:


#Using GridSearchCV to find the best algorithm for this problem

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB


# In[23]:


# Creating a function to calculate best model for this problem

def find_best_model(X, y):
    models = {
        'logistic_regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto'),
            'parameters': {
                'C': [1,5,10]
               }
        },
        
        'random_forest': {
            'model': RandomForestClassifier(criterion='gini'),
            'parameters': {
                'n_estimators': [10,15,20,50,100,200]
            }
        },
        
        'Naive Bayes': {
            'model' : BernoulliNB(),
            'parameters': {
                'alpha' : [0,1.0]
            }
        }
    }

    scores = [] 
    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
        
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        
    return pd.DataFrame(scores, columns=['model','best_parameters','score'])

#find_best_model(X_train, y_train)


# In[24]:


find_best_model(X_train, y_train)


# In[22]:


# Using cross_val_score for gaining average accuracy

from sklearn.model_selection import cross_val_score
scores = cross_val_score(RandomForestClassifier(n_estimators=20, random_state=0), X_train, y_train, cv=10)
print('Average Accuracy : {}%'.format(round(sum(scores)*100/len(scores)), 3))


# In[17]:


# Creating Random Forest Model

rf = RandomForestClassifier(n_estimators=100, random_state=69)
rf.fit(X_train, y_train)


# ## Model Evaluation 

# In[18]:


# Creating a confusion matrix

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, cbar=False, annot=True)
plt.show()


# In[19]:


# Classification Report

print(classification_report(y_test, y_pred))


# In[20]:


feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


# In[21]:


feature_importances


# In[29]:





# In[ ]:




