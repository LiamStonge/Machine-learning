#   Logistic Regression
#importing the libraries
import numpy as np #used for math in python
import matplotlib.pyplot as plt
import pandas as pd #used to import dataset

#importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

#create an array to contain independent variables and one to contain dependent variables
X = dataset.iloc[:, [2, 3]].values #independent
Y = dataset.iloc[:, -1].values #array for dependent (-1 means the last index of array)
#splitting the dataset into the training set and test set
#this will see if the machine can learn from the train set to prodict the correct values of the test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
# we need to transform variables so that they have their variables in the same scale.
# this will make sure that the machine doesn't override smaller variables, for example in this dataset the age is much smaller than the salary.
#   Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the classifier to the training set
#create your classifier here

# Predicting the Test set results!
y_pred = classifier.predict(X_test) #this predicts the test results for x test

# making the confusion matrix (contains the correct and incorrect prediction of y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

# visualising the training set results 
from matplotlib.colors import ListedColormap
X_set, Y_set, = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start= X_set[:, 0].min()-1, stop = X_set[:, 0].max()+1, step = 0.01),
                     np.arange(start= X_set[:, 1].min()-1, stop = X_set[:, 1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.legend()
plt.show()

# visualising the Test set results 
from matplotlib.colors import ListedColormap
X_set, Y_set, = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start= X_set[:, 0].min()-1, stop = X_set[:, 0].max()+1, step = 0.01),
                     np.arange(start= X_set[:, 1].min()-1, stop = X_set[:, 1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.legend()
plt.show()
