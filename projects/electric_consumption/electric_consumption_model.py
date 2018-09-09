import model_manager
import datetime_utils
import data_processor

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
data = data_processor.GetData()
X = data[0]
Y = data[1]
split_point = int(X.shape[0]*0.8)
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

model = Sequential()
model.add(Dense(10, input_dim= Xtrain.shape[1],activation='softplus'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
model.fit(Xtrain, Ytrain, epochs=10)

model_manager.SaveElectricConsumptionModel(model)
