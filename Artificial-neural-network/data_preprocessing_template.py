#Data Preprocessing 

#importing the libraries
import numpy as np #used for math in python
import matplotlib.pyplot as plt
import pandas as pd #used to import dataset

#importing the dataset
dataset = pd.read_csv('Data.csv')

#create an array to contain independent variables and one to contain dependent variables
X = dataset.iloc[:, :-1].values #independent
Y = dataset.iloc[:, -1].values #array for dependent (-1 means the last index of array)

#Taking care of missing data in the csv file (data exel file)
from sklearn.preprocessing import Imputer #library to preprocess datasets
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 1:3])#fit in the matrix x the rigth colouums (the ones with missing data)
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #the label encoder changes the string variables into numerical variables so the machine can calculate 
onehotencoder = OneHotEncoder(categorical_features = [0]) #creates 3 categories for the countries (to remove 0,1,2 so that the machine doesn't think they are weigths)
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder() #does the same thing for the dependent variable (changes No/Yes into 0/1)
Y = labelencoder_Y.fit_transform(Y)

#splitting the dataset into the training set and test set
#this will see if the machine can learn from the train set to prodict the correct values of the test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# we need to transform variables so that they have their variables in the same scale.
# this will make sure that the machine doesn't override smaller variables, for example in this dataset the age is much smaller than the salary.
#   Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)















