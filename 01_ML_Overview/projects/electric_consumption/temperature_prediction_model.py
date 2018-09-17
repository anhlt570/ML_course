import datetime_utils
import numpy
import pandas
import os
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt


def SaveModel(model):
    if not os.path.exists('models'):
        os.mkdir('models')
    model.save('models/temperature_prediction.h5')


def LoadModel():
    model_path = 'models/temperature_prediction.h5'
    if not os.path.exists(model_path):
        return -1
    return load_model(model_path)


temperature_data = pandas.read_csv('data/nyc_weather.csv').values
Y = [i for i in temperature_data[:, 1]]
X = []
for i in range(0, temperature_data.shape[0]):
    date = datetime_utils.GetTime(temperature_data[i][0])
    month = date.month
    day_of_week = datetime_utils.GetDayOfWeek(date)
    hour = date.hour
    X.append([month, day_of_week, hour])
categorical_month = to_categorical([X[i][0] for i in range(0, len(X))])
categorical_day = to_categorical([X[i][1] for i in range(0, len(X))])
categorical_hour = to_categorical([X[i][2] for i in range(0, len(X))])
for i in range(0, len(X)):
    X[i] = categorical_month[i].tolist() + \
        categorical_day[i].tolist()+categorical_hour[i].tolist()

X = numpy.array(X)
split_point = int(len(X) * 0.8)

Xtrain = X[:split_point]
Xtest = X[split_point:]
Ytrain = Y[:split_point]
Ytest = Y[split_point:]

model = Sequential()
model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='softplus'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
model.fit(Xtrain, Ytrain)
print('score = ', model.evaluate(Xtest, Ytest))

plt.plot(range(0, Xtest.shape[0]), Ytest, color='red')
plt.plot(range(0, Xtest.shape[0]), model.predict(Xtest), color='green')
plt.show()