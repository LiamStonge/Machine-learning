
"""
Created on Fri Nov  3 16:02:01 2017

@author: liams
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from keras.models import Sequential 
from keras.layers import Dense, LSTM , Dropout
from pandas_datareader import data
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
from pandas_datareader._utils import RemoteDataError


# importing the training set

prediction_days = 20
company_symbol= input("Enter the companies stock symbol: ")
x_train = []
y_train = []
regressor = Sequential()
dimension=1;
timesteps=int(input("Enter the number of timesteps you wish to use: "))

# Fuction that retrieves the data from given company symbol and dates
def data_retriever(symbol, start_date, end_date=None):
    start = datetime.datetime(start_date[0],start_date[1],start_date[2])# date = year, month, day
    if not end_date:
        end = datetime.datetime.now()
    else:
        end =datetime.datetime(end_date[0],end_date[1],end_date[2])
        
    stocks_data = data.DataReader(symbol, "yahoo", start, end)
    return stocks_data

# Creating a data stucute with 60 timesteps and 1 output
# 60 timesteps means that the RNN will look at the past 60 stock prices and keep it in its memory (3 months)
# the output will be the stock price at time t+1
def data_training_set(x_train, y_train, timesteps, training_set, dimension, test_days):
    for i in range(timesteps, (len(training_set))-test_days):
        x_train.append(training_set[i-timesteps: i, 0:dimension])
        y_train.append(training_set[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], dimension))
    return x_train, y_train

# Creat the class to handle all needed functions for the RNN model
class RNN_model: 
    model = Sequential()
    load_model_has_been_called = False

    def __init__(self, x_train, dimension):
        # Adding the first LSTM layer and some dropout regularisation
        self.model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], dimension) ))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 50, return_sequences = True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 50, return_sequences = True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 50))
        self.model.add(Dropout(0.2))
        # Adding the output layer
        self.model.add(Dense(units = 1))
        return None
    # Compile RNN
    def compile_model(self, str_optimizer, str_loss):    
        self.model.compile(optimizer = str_optimizer, loss = str_loss)
        return None
	# Train the RNN
    def train_model(self, x_train, y_train, epochs, batch_size):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        return None
    def predict_data(self, x_test):
        return self.model.predict(x_test)
    def save_model(self, symbol, timesteps):
        model_json = self.model.to_json()
        with open("Models/model."+symbol+"."+str(timesteps)+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("Models/model."+symbol+"."+str(timesteps)+".h5")
        print("Saved model to disk")
        return None
    def load_model(self, symbol, timesteps):
        self.load_model_has_been_called = True
        json_file = open("Models/model."+symbol+"."+str(timesteps)+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("Models/model."+symbol+"."+str(timesteps)+".h5")
        print("Loaded model from disk")

                
   
    
# Fuction to get the predicted and real stock price and show on graph the accuracy
def simple_graph(real_stock_price, predicted_stock_price, company_symbol):
    plt.plot(real_stock_price, color = 'red', label = ('real '+company_symbol+' Stock Price'))
    plt.plot(predicted_stock_price, color = 'blue', label = ('Predicted '+company_symbol+' Stock Price'))
    plt.title(company_symbol+' Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
# Test the data on the test set 
def test_rnn(model_rnn, training_set, prediction_days, timesteps, dimension):
    test_stock = training_set[len(training_set)-prediction_days:, 0:1]
    real_stock_price = test_stock
    next_day_real_prices = real_stock_price[1:len(real_stock_price) ]
    
    # Getting the predicted stock price of 2017
    inputs = training_set[len(training_set) - len(test_stock) - (timesteps):]
    inputs = inputs.reshape(-1,dimension)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(timesteps, (timesteps+prediction_days)):
        X_test.append(inputs[i-timesteps:i, 0:dimension])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], dimension))

    predicted_stock_price = model_rnn.predict_data(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    simple_graph(next_day_real_prices, predicted_stock_price, company_symbol)


# Sometimes there is an error when getting the data from yahoo, i wasn't sure why but it only happened on occasion and simply
# restarting would fix it so i added an error check 
while True:
  try:
    dataset = data_retriever(company_symbol, start_date = (2010,1,1))
    break
  except RemoteDataError:
      print("error downloading data")


training_set = dataset.iloc[:, 0:dimension].values
# Feature Scaling
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


# Create the data for the training
x_train, y_train =  data_training_set(x_train, y_train, timesteps, training_set_scaled, dimension, prediction_days)


# Creating the RNN 
model_rnn=RNN_model(x_train, dimension)
# Check to see if you already have a trained model saved
if(input("Do you have a saved model? (y/n) ") == "y"):
    model_rnn.load_model(company_symbol, timesteps)
else: # If you don't then train a new one
    model_rnn.compile_model('adam', 'mean_squared_error')
    model_rnn.train_model(x_train, y_train, int(input("Enter num of epochs: ")), 32)
# Ask to see if you would like to test your model on his prediction of trends
if(input("Do you wish to test your RNN? (y/n): ") =="y"):
    test_rnn(model_rnn, training_set, prediction_days, timesteps, dimension)
# If you model is new, then ask to see if you wish to save it and its weights 
if(model_rnn.load_model_has_been_called==False):
    if(input("Do you wish to save model? (y/n): ")=="y"):
            model_rnn.save_model(company_symbol, timesteps)
