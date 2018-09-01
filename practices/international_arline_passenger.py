import pandas
import numpy
from sklearn.neural_network import MLPRegressor


csv_data = pandas.read_csv("data/international_airline_passengers.csv")
data = csv_data.values
times = data[:,0]
X = numpy.zeros(shape =(times.shape[0],2))
Y = data[:,1]
for i in range(times.shape[0]):
    t = times[i].split('-')
    X[i] = [int(s) for s in t]
Xtrain = X[:-2]
Ytrain = Y[:-2]
Xtest = X[-2:]
Ytest = Y[-2:]

model = MLPRegressor(hidden_layer_sizes=50)
model.fit(Xtrain,Ytrain)
print ("Predict: ",model.predict(Xtest))
print ("Validation data: ", Ytest)
print("Score: ", model.score(Xtest,Ytest))
print("this is how the world end")