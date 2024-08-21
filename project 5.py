#!/usr/bin/env python
# coding: utf-8

# # CREDIT CARD FRAUD DETECTION
# 
# Build a machine learning model to identify fraudulent credit card transactions.Preprocess and normalize the transaction data, handle class imbalance issues, and split the dataset into training and testing sets.Train a classification algorithm, such as logistic regression or random forests, to classify transactions as fraudulent or genuine.
# Evaluate the model's performance using metrics like precision, recall,and F1-score, and consider techniques like oversampling or
# undersampling for improving results.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv('creditcard.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


# Preprocess and normalize the data
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df.head()


# In[8]:


# Scatter plots for numerical features vs. target
numerical_features = ['Time', 'Amount']
for feature in numerical_features:
    sns.scatterplot(data=df, x=feature, y='Class')
    plt.title(f'Scatter Plot of Class vs. {feature}')
    plt.show()


# In[9]:


# Box plot for the 'Amount' feature
sns.boxplot(data=df, x='Class', y='Amount')
plt.title('Box Plot of Amount by Class')
plt.show()


# In[10]:


# Histogram of the 'Amount' feature
sns.histplot(df['Amount'], kde=True, bins=30)
plt.title('Histogram of Amount')
plt.show()


# In[11]:


# Pair plot of selected features
selected_features = ['Time', 'Amount', 'Class']
sns.pairplot(df[selected_features], hue='Class', palette='coolwarm')
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()


# In[3]:


# Separate the majority and minority classes
df_majority = df[df['Class'] == 0]
df_minority = df[df['Class'] == 1]

# Upsample the minority class
from sklearn.utils import resample
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=42) # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
print(df_upsampled['Class'].value_counts())


# In[4]:


#Split the Dataset into Training and Testing Sets
# Define features and target variable
X = df_upsampled.drop('Class', axis=1)
y = df_upsampled['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


#Train a Classification Algorithm
# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# In[6]:


#Evaluate the Model's Performance
# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualize the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[7]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# In[9]:


# Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[ ]:





# In[ ]:




