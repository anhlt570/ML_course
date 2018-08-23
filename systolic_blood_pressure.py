import numpy
import pandas
import sklearn.linear_model

excel_data = pandas.read_excel("data/systolic blood presure.xls")
data = excel_data.values
X = data[ : , 1 : ]
Y = data[ : , 0]

Xtrain = X[ : int(X.shape[0] * 0.7)]
Ytrain = Y[ : int(Y.shape[0] * 0.7)]

Xtest = X[ int(X.shape[0] * 0.7): ]
Ytest = Y[ int(Y.shape[0] * 0.7): ]

model = sklearn.linear_model.LinearRegression()
model.fit(Xtrain,Ytrain)
print(model.score(Xtest,Ytest))
print(model.predict(Xtest))