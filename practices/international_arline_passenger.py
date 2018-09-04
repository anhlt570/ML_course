import pandas
import numpy
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

def model_by_sklearn(Xtrain, Ytrain):
    model = MLPRegressor(hidden_layer_sizes=5000)
    model.fit(Xtrain,Ytrain)
    return model

def model_by_keras(Xtrain, Ytrain):
    model = Sequential()
    model.add(Dense(5000, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='linear'))

    adam = Adam(lr=0.01)
    model.compile(optimizer=adam, loss='mse')
    model.fit(Xtrain, Ytrain, epochs=1000)
    return model

csv_data = pandas.read_csv("data/international_airline_passengers.csv")
data = csv_data.values
times = data[:,0]
X = numpy.zeros(shape =(times.shape[0],2))
Y = data[:,1]
for i in range(times.shape[0]):
    t = times[i].split('-')
    X[i] = [int(s) for s in t]
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

Xtrain = X[:-2]
Ytrain = Y[:-2]
Xtest = X[-2:]
Ytest = Y[-2:]

keras_model = model_by_keras(Xtrain, Ytrain)
sklearn_model = model_by_sklearn(Xtrain, Ytrain)
keras_result = numpy.append(Ytrain, model_by_keras(Xtrain, Ytrain).predict(Xtest))
sklearn_result = numpy.append(Ytrain, model_by_sklearn(Xtrain, Ytrain).predict(Xtest))

print('real data= ',Ytest)
print('keras result= ',keras_result[-2:])
print('sklearn result= ',sklearn_result[-2:])

plt.plot(range(0,X.shape[0]), Y, label='reality',color='r')
plt.plot(range(0, X.shape[0]), sklearn_result, label='sklearn',linestyle='dotted', color='b')
plt.plot(range(0, X.shape[0]), keras_result, label='keras',linestyle='dotted', color='g')
plt.legend(loc='upper left')
plt.show()





