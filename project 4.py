#!/usr/bin/env python
# coding: utf-8

# # SALES PREDICTION USING PYTHON
# 
# Sales prediction involves forecasting the amount of a product that
# customers will purchase, taking into account various factors such as
# advertising expenditure, target audience segmentation, and
# advertising platform selection.
# In businesses that offer products or services, the role of a Data
# Scientist is crucial for predicting future sales. They utilize machine
# learning techniques in Python to analyze and interpret data, allowing
# them to make informed decisions regarding advertising costs. By
# leveraging these predictions, businesses can optimize their
# advertising strategies and maximize sales potential. Let's embark on 
# the journey of sales prediction using machine learning in Python.

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


df = pd.read_csv('advertising.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[9]:


# Scatter plot of sales vs. TV advertising
sns.scatterplot(data=df, x='TV', y='Sales')
plt.title('Scatter Plot of Sales vs. TV Advertising')
plt.show()

# Scatter plot of sales vs. Radio advertising
sns.scatterplot(data=df, x='Radio', y='Sales')
plt.title('Scatter Plot of Sales vs. Radio Advertising')
plt.show()

# Scatter plot of sales vs. Newspaper advertising
sns.scatterplot(data=df, x='Newspaper', y='Sales')
plt.title('Scatter Plot of Sales vs. Newspaper Advertising')
plt.show()


# In[10]:


# Box plot of sales
sns.boxplot(data=df, y='Sales')
plt.title('Box Plot of Sales')
plt.show()


# In[11]:


# Histogram of sales
sns.histplot(df['Sales'], kde=True)
plt.title('Histogram of Sales')
plt.show()


# In[12]:


# Define features and target variable
X = df[['TV', 'Radio', 'Newspaper']]  # Features
Y = df['Sales']  # Target variable

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)


# In[13]:


# Make predictions
Y_pred = model.predict(X_test)
Y_pred


# In[16]:


# Calculate evaluation metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[17]:


# Calculate MAPE (Mean Absolute Percentage Error)
mape = (np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100)

# Calculate accuracy within a certain threshold
threshold = 0.10  # 10% threshold
accurate_predictions = np.mean(np.abs((Y_test - Y_pred) / Y_test) < threshold) * 100

print(f'Mean Absolute Percentage Error: {mape}%')
print(f'Accuracy within {threshold*100}% threshold: {accurate_predictions}%')


# In[18]:


# Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[19]:


sns.pairplot(df)
plt.suptitle('Pair Plot of TV, Radio, Newspaper, and Sales', y=1.02)
plt.show()


# In[ ]:




