#!/usr/bin/env python
# coding: utf-8

# # MOVIE RATING PREDICTION WITH PYTHON
# Build a model that predicts the rating of a movie based on
# features like genre, director, and actors. You can use regression
# techniques to tackle this problem.
# The goal is to analyze historical movie data and develop a model
# that accurately estimates the rating given to a movie by users or
# critics.
# Movie Rating Prediction project enables you to explore data
# analysis, preprocessing, feature engineering, and machine
# learning modeling techniques. It provides insights into the factors
# that influence movie ratings and allows you to build a model that
# can estimate the ratings of movies accurately.

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[15]:


# Load dataset
df = pd.read_csv('movies.csv',encoding='latin1')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


# Handling missing values
df = df.dropna(subset=['Votes', 'Genre', 'Director', 'Rating'])
df = df.dropna(subset=['Duration', 'Actor 1', 'Actor 2', 'Actor 3'])


# In[9]:


df.isnull().sum()


# In[11]:


# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Votes', y='Rating', data=df)
plt.title('Votes vs. Rating')
plt.show()


# In[7]:


# Pair plot
sns.pairplot(df[['Votes', 'Duration', 'Rating', 'Genre','Director','Actor 1', 'Actor 2', 'Actor 3']])
plt.show()


# In[15]:


# Subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
sns.scatterplot(x='Votes', y='Rating', data=df, ax=axs[0, 0])
sns.scatterplot(x='Duration', y='Rating', data=df, ax=axs[0, 1])
sns.boxplot(x='Genre', y='Rating', data=df, ax=axs[1, 0])
sns.histplot(df['Rating'], bins=30, kde=True, ax=axs[1, 1])
plt.show()


# In[ ]:


# Facet Grid
g = sns.FacetGrid(df, col="Genre", col_wrap=4, height=4)
g.map(sns.scatterplot, 'Votes', 'Rating')
plt.show()


# In[4]:


# Box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Rating', data=df)
plt.title('Ratings by Genre')
plt.show()


# In[3]:


# Histogram
plt.figure(figsize=(12, 6))
sns.histplot(df['Rating'], bins=30, kde=True)
plt.title('Distribution of Movie Ratings')
plt.show()


# In[11]:


#Feature Engineering
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
df_encoded.head()


# In[19]:


df_encoded['Votes'] = pd.to_numeric(df_encoded['Votes'], errors='coerce')
df_encoded = df_encoded.dropna(subset=['Votes'])
df_encoded['log_votes'] = df_encoded['Votes'].apply(lambda x: np.log1p(x))
print(df_encoded[['Votes', 'log_votes']].head())


# In[16]:


# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Evaluate Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Mean Squared Error: {mse_rf}')
print(f'Random Forest Mean Absolute Error: {mae_rf}')
print(f'Random Forest R-squared: {r2_rf}')


# In[18]:


# Splitting the dataset into training and testing sets
X = df.drop('Rating', axis=1)
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)


# In[ ]:





# In[10]:




