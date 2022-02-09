#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Model
# 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This code is the implementation of linear regression model for yield 
# predication.
# 
# We are using linear regression model which is imported from Sklearn library.
# 
# Data is comprised of 234 samples. Each training sample is consist of 16 input
# features(x1,x2,x3,....) and corresponding output(y) i.e yield.
# 
# x1= Flied 
# x2= Seed
# x3= max temp
# x4= min temp
# x5= Wind speed
# x6= humidity
# x7= Precipitation
# x8= Rain fall
# x9= N
# x10=P
# x11=K
# x12=NDVI
# x13=NDMI
# x14=MSAVI
# x15=NRDE
# x16= Soil fertility
# 
# y= Yield 
# 
# linear model
# h=theta0+ theta1*x1+ theta2*x2+ theta3*x3 .......theta16*x16
# 
# these theta's are trained during training.
# 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 1. Importing required packages.
# 2. Read the data/load the data.
# 3. Data Cleaning.
# 4. Visualizing the data, how data looks like.
# 5. Spliting dataset into training and testing data.
# 6. Imported Linear regression model is Trained using training data
# 7. Predication is performed by trained model using testing data
# 8. Evaluate the Model: Measure the Mean Sqaure Error
# 9. Prediction for the new sample
# 
# 

# # Importing required packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error
from sklearn.metrics import  mean_absolute_error


# #  Read the data/load the data

# In[2]:


# we are using pandas libaray for reading the file
# We are reading ecxel file and path should be where you ecxel file is located.
# place "r" before the path string to address special character,such as '\'.
# Don't forget to put the file name at the end of the path + '.xlsx'

data=pd.read_excel(r"C:\python\Yield Predicition\plain 1.xlsx")


# In[3]:


#Showing the first five entries of data
data.head()


# # Data Cleaning

# In[4]:


# As there some extra columns so we need to perform data cleaning

data.drop(columns='Month', inplace=True)
data.head()


# # Visualizing the data, how data looks like.

# In[5]:


# Information about your data, how many entries and which type of data is present in dataframe
data.info()


# In[6]:


#tells the shape of dataframe
data.shape


# In[7]:


data.plot(kind='scatter', x='Rainfall', y='Yield', figsize=(6,6))


# In[8]:


data.plot(kind='scatter', x='Precipitation (mm)', y='Yield', figsize=(6,6))


# In[9]:


data.plot(kind='scatter', x='Humidity (%)', y='Yield', figsize=(6,6))


# # Spliting dataset into training and testing data
# 
# Total of 60 samples we are taking first 190 samples as training data and last 44 samples as testing data

# In[10]:


#Training data
training_data=data.iloc[:190, 0:15]
training_data.head()


# In[11]:


training_data.shape


# In[12]:


training_data.describe()


# In[13]:


training_label=data.iloc[:190, -1]
training_label.head()


# In[14]:


print('shape:', training_label.shape)
print('type:', training_label.dtype)


# In[15]:


#testing data

testing_data=data.iloc[190:, 0:15]
testing_data.head()


# In[16]:


testing_label=data.iloc[190:, -1]
testing_label.head()


# # Imported Linear regression model is Trained using training data

# In[17]:


# linear regression model from sklearn is used
# normalization is performed
model = linear_model.LinearRegression(normalize=True)

#model is trained using our data
model.fit(training_data,training_label)
y_train_pred = model.predict(training_data)


# # Predication is performed by trained model using testing data

# In[18]:


# Now prediction is formed using trained model
y_predicted = model.predict(testing_data)


# In[19]:


# Checking the importance of features
importance = model.coef_

# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance

plt.bar([x for x in range(len(importance))], importance)
plt.show()


# # Evaluate the Model: Measure the Mean Sqaure Error

# In[20]:


# Model is evaluated by measuring Mean squared error
print("Mean squared error is: ", mean_squared_error(testing_label, y_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)


# In[21]:



error= mean_absolute_error(testing_label, y_predicted)
print('MAE: %.3f' % error)


# In[22]:


plt.plot(training_label, y_train_pred,'*r')
plt.plot(testing_label, y_predicted, '*g')
plt.figure()


# #  Prediction for the new sample

# In[23]:


#Just for checking I have taken testing sample


# In[24]:


print('input features')
testing_data.iloc[3]


# In[25]:


print('Yield')
testing_label.iloc[3]


# In[26]:


# Here you can enter new input in array below


# In[27]:


test_sample= np.array([8,2,33,20.20,1.6,58,0,11.50,99,31,32,0.51,0.36,0.32,0.09])
test_sample


# In[28]:


test_sample.shape


# In[29]:


type(test_sample)


# In[30]:


test_sample.ndim


# In[31]:


test_sample=test_sample.reshape(1,15)
print(test_sample.shape)
print(test_sample.ndim)


# In[32]:


y_predicted = model.predict(test_sample)


# In[33]:


y_predicted


# In[ ]:




