#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER CLASSIFICATION
# 
# The Iris flower dataset consists of three species: setosa, versicolor,
# and virginica. These species can be distinguished based on their
# measurements. Now, imagine that you have the measurements
# of Iris flowers categorized by their respective species. Your
# objective is to train a machine learning model that can learn from
# these measurements and accurately classify the Iris flowers into
# their respective species.
# 
# Use the Iris dataset to develop a model that can classify iris
# flowers into different species based on their sepal and petal
# measurements. This dataset is widely used for introductory
# classification tasks.

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


iris = load_iris()
X = iris.data  
y = iris.target


# In[4]:


iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
print(iris_df.head())


# In[5]:


iris_df.head()


# In[6]:


iris_df.describe()


# In[7]:


iris_df.info()


# In[8]:


iris_df.isnull().sum()


# In[9]:


# Boxplots of features by species
plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=iris_df.iloc[:, i], data=iris_df)
    plt.title(iris.feature_names[i])
plt.tight_layout()
plt.show()


# In[10]:


# Pairplot to visualize relationships between features
sns.pairplot(iris_df, hue='species', palette='Set1')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()


# In[11]:


# Histogram
plt.figure(figsize=(10, 6))
for i in range(3):
    sns.histplot(iris_df[iris_df['species'] == i]['petal width (cm)'], kde=True, label=iris.target_names[i])
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Width by Species')
plt.legend()
plt.show()


# In[12]:


# Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris_df)
plt.title('Scatterplot of Sepal Length vs Petal Length by Species')
plt.show()


# In[13]:


#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


#Choose a model and train it
model = SVC(kernel='linear', C=1, gamma='auto')
model.fit(X_train, y_train)


# In[15]:


#Evaluate the model
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[16]:


#Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[17]:


# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# In[19]:


#Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(iris_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


# In[21]:


#Make predictions
y_pred = model.predict(X_test)
print(y_pred)


# In[24]:


#Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[ ]:




