# Artificial Neural Network


# Part 1 - data preprocessing 
#importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd #used to import dataset

#importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#create an array to contain independent variables and one to contain dependent variables
X = dataset.iloc[:, 3:13].values #independent
Y = dataset.iloc[:, -1].values #array for dependent 
#encoding the string variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #the label encoder changes the string variables into numerical variables so the machine can calculate 
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) #creates 3 categories for the countries (to remove 0,1,2 so that the machine doesn't think they are weigths)
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] 



#splitting the dataset into the training set and test set
#this will see if the machine can learn from the train set to prodict the correct values of the test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)




# Part 2 - Making the ANN
# importing the Keras librares and packages
import keras
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
# arguments in order - number of hidden Nueron , randomly initializes the weigth , chooses the formual for the activation sum, number of inputs 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))#first hidden layer and the input layer
classifier.add(Dropout(rate=0.1))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))#second hidden layer
classifier.add(Dropout(rate=0.1))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))#output layer
#arguments in order - adam is a stoicastic formula ,  choose the loss function based on the amount of categories you have  , the accuracy will increase until reaching top accuracy 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])#compile ANN

# fitting the ANN to the training set
classifier.fit(X_train, Y_train, batch_size = 25, epochs = 100)

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results!
y_pred = classifier.predict(X_test) #this predicts the test results for x test
y_pred = (y_pred > 0.5)
# making the confusion matrix (contains the correct and incorrect prediction of y_pred)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

# Part 4 - evaluating, improving and tuning the ANN

# Evaluating ANN this is to check what the real accuracy and to check were we are in the var-bios trade off (we want high accuracy and small variance) 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X=X_train, y=Y_train, cv=10, n_jobs = 1)

mean = accuracies.mean()
variance = accuracies.std()

#improving the ANN
# Dropout regularisation to reduce overfitting if needed
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32], 
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_








