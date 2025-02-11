# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Processing
data_file = pd.read_csv('C:/Users/SARVESH/Desktop/Dataplay/Excel-Formulas-and-Functions-Complete-Sheet.csv')

# Printing the first 5 rows of dataframe
print(data_file.head())

# Number of rows and coloumn
print(data_file.shape)

# Gathering some more information of the data
print(data_file.info())

# Number of missing values
print(data_file.isnull().sum())

# Droping Cabin coloumn
data_file = data_file.drop(columns= 'Cabin', axis= 1)
print(data_file)

# Replacing the missing values in Age coloumn with the mean value
data_file['Age'].fillna(data_file['Age'].mean(), inplace= True)
print(data_file)

# Finding the mode value of Embarked coloumn
print(data_file['Embarked'].mode())
print(data_file['Embarked'].mode()[0])

# Replacing the missing values of Embarked coloumn with the mode value
data_file['Embarked'].fillna(data_file['Embarked'].mode()[0], inplace= True)
print(data_file)

# Checking the missing values
print(data_file.isnull().sum())

# Getting some statistical measures of the data
print(data_file.describe())

# Finding the number of people survived and not survived
print(data_file['Survived'].value_counts())

# Finding the number of people survived and not survived based on gender
print(data_file['Sex'].value_counts())

# Finding the number of people survived and not survived based on Embarked
print(data_file['Embarked'].value_counts())

# Data Visualization
sns.set()

# Count plot for Survived Column
plt.figure(figsize= (5,3))
sns.countplot(x= 'Survived', data= data_file)

# Count plot for Sex colomn
plt.figure(figsize= (5,3))
sns.countplot(x= 'Sex', data= data_file)

# Count plot of survived gender wise
plt.figure(figsize= (5,3))
sns.countplot(x= 'Sex', hue= 'Survived', data= data_file)

# Count plot for Pclass column
plt.figure(figsize= (5,3))
sns.countplot(x= 'Pclass', data= data_file)

# Count plot of survived gender wise
plt.figure(figsize= (5,3))
sns.countplot(x= 'Pclass', hue= 'Survived', data= data_file)
plt.show()

# Converting the catagorical column
data_file.replace({'Sex': {'male':0, 'female':1}, 'Embarked': {'S':0, 'C':1, 'Q':2}}, inplace= True)
print(data_file)

# Separating Features and Targets
X = data_file.drop(columns= ['PassengerId', 'Survived', 'Ticket', 'Name'], axis= 1)
Y = data_file['Survived']
print(X)
print(Y)

# Splitting the data into Training data and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state= 2)
print(X.shape, X_train.shape, X_train.shape)

# Model Training
model = LogisticRegression()

# Training the Logistic Regression Model with data
model.fit(X_train, Y_train)
print(model)

# Model Evaluation on Training data
X_train_prediction = model.predict(X_train)
print(X_train_prediction)

# Accuracy Scoring on Training data
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Accuracy Score of Training data:", training_data_accuracy)

# Model Evaluation on Test data
X_test_prediction = model.predict(X_test)
print(X_test_prediction)

# Accuracy Scoring on Test data
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy Score of Test data:", test_data_accuracy)