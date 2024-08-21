#!/usr/bin/env python
# coding: utf-8

# # TITANIC SURVIVAL PREDICTION
# 
# Use the Titanic dataset to build a model that predicts whether a
# passenger on the Titanic survived or not. This is a classic beginner
# project with readily available data.
# The dataset typically used for this project contains information
# about individual passengers, such as their age, gender, ticket
# class, fare, cabin, and whether or not they survived.

# In[140]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[141]:


# Load the Titanic dataset from Seaborn
titanic = sns.load_dataset('titanic')


# In[142]:


# Display the first few rows of the dataset
print(titanic.head())


# In[143]:


titanic.shape


# In[144]:


titanic.info()


# In[145]:


titanic.describe()


# In[146]:


titanic.isnull().sum()


# In[151]:


# Fill missing values in 'age' with median
titanic['age'].fillna(titanic['age'].median(), inplace=True)


# In[154]:


# Remove rows with null values
titanic_cleaned = df.dropna()

# Drop unnecessary columns
titanic_cleaned = titanic_cleaned.drop(columns=['who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'])


# In[156]:


titanic_cleaned.isnull().sum()


# In[107]:


# Visualize the number of survivors
sns.countplot(x='survived', data=titanic)
plt.title('Count of Survivors (1 = Survived, 0 = Not Survived)')
plt.show()


# In[108]:


# Visualize survival by sex
sns.countplot(x='survived', hue='sex', data=titanic)
plt.title('Survival by Sex (1 = Survived, 0 = Not Survived)')
plt.show()


# In[109]:


# Scatter Plot: Age vs Fare colored by Survival
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='fare', hue='survived', data=titanic)
plt.title('Scatter Plot: Age vs Fare')
plt.show()


# In[110]:


# Pair Plot: Pair plot of features colored by Survival
sns.pairplot(titanic[['survived', 'age', 'fare', 'pclass', 'sibsp', 'parch']], hue='survived')
plt.show()


# In[111]:


# Sub Plot: Survived count for each Pclass
plt.figure(figsize=(10, 6))
sns.countplot(x='pclass', hue='survived', data=titanic)
plt.title('Survival Count by Passenger Class')
plt.show()


# In[112]:


# FacetGrid: Age distribution by Pclass and Survival
g = sns.FacetGrid(titanic, col='pclass', hue='survived', height=4, aspect=1.5)
g.map(sns.histplot, 'age', kde=True).add_legend()
plt.show()


# In[113]:


# Box Plot: Fare distribution by Pclass and Survival
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='fare', hue='survived', data=titanic)
plt.title('Fare Distribution by Passenger Class and Survival')
plt.show()


# In[114]:


# Convert categorical variables to numerical
titanic['sex'] = pd.factorize(titanic['sex'])[0]
titanic['embarked'] = pd.factorize(titanic['embarked'])[0]


# In[115]:


# Select features and target variable
X = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = titanic['survived']


# In[116]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[117]:


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)


# In[118]:


# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[119]:


# Classification report
print(classification_report(y_test, y_pred))


# In[120]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[121]:


# Example prediction (replace with actual data)
new_data = pd.DataFrame([[3, 1, 25, 0, 0, 7.75, 2]], columns=X.columns)
prediction = rf_classifier.predict(new_data)
print(f'Prediction: {prediction}')


# In[127]:


df = sns.load_dataset('titanic')


# In[128]:


sns.boxplot(x='pclass',data=df)


# In[94]:


pwd


# In[ ]:





# In[ ]:





# In[ ]:




