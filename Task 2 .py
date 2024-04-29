#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Step 1: Data Loading
import pandas as pd
file_path = "C:\\Users\\HP\\Desktop\\IRIS.csv"
df = pd.read_csv(file_path)


# In[3]:


# Step 2: Data Exploration
print("First few rows of the dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nTarget variable distribution:")
print(df['species'].value_counts())


# In[4]:


# Step 3: Data Preprocessing
# No preprocessing is required in this case since the data is already provided in a suitable format.


# In[5]:


# Step 4: Feature Selection
# Selecting features for classification
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']


# In[6]:


# Step 5: Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# In[7]:


# Step 6: Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predicting on the test set
y_pred = clf.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")

# Generating classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generating confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[ ]:




