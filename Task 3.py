#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset (using raw string literal)
data = pd.read_csv(r"C:\Users\HP\Desktop\advertising.csv")


# In[4]:


# Step 2: Data Exploration
print("Dataset info:")
print(data.info())

print("\nSummary statistics:")
print(data.describe())


# In[ ]:


# Step 3: Data Preprocessing
# No preprocessing required for this example


# In[5]:


# Step 4: Split the data into features (X) and target variable (y)
X = data.drop('Sales', axis=1)
y = data['Sales']


# In[6]:


# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Step 6: Model Selection & Training
model = LinearRegression()
model.fit(X_train, y_train)


# In[8]:


# Step 7: Model Evaluation

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")


# In[9]:


# Step 8: Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()


# In[ ]:




