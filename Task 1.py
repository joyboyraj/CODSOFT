#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


titanic_df = pd.read_csv(r'C:\Users\HP\Desktop\Titanic-Dataset.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(titanic_df.head())






# In[10]:


# Basic information about the dataset
print("\nDataset info:")
print(titanic_df.info())


# In[11]:


# Summary statistics
print("\nSummary statistics:")
print(titanic_df.describe())


# In[12]:


# Check for missing values
print("\nMissing values:")
print(titanic_df.isnull().sum())


# In[13]:


# Data visualization
# Example: Survival count
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=titanic_df)
plt.title('Survival Count')
plt.show()


# In[14]:


# Age distribution by Survived
plt.figure(figsize=(10, 6))
sns.histplot(x='Age', data=titanic_df, hue='Survived', bins=30, kde=True)
plt.title('Age Distribution by Survived')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(['Did not survive', 'Survived'])
plt.show()


# In[15]:


# Survival by Sex
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', hue='Survived', data=titanic_df)
plt.title('Survival by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(['Did not survive', 'Survived'])
plt.show()


# In[16]:


# Survival by Passenger Class
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=titanic_df)
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(['Did not survive', 'Survived'])
plt.show()


# In[17]:


# Survival by Embarked Port
plt.figure(figsize=(8, 6))
sns.countplot(x='Embarked', hue='Survived', data=titanic_df)
plt.title('Survival by Embarked Port')
plt.xlabel('Embarked Port')
plt.ylabel('Count')
plt.legend(['Did not survive', 'Survived'])
plt.show()


# In[ ]:





