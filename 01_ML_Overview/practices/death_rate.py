import numpy
import pandas
from sklearn.linear_model import LinearRegression

excel_data = pandas.read_excel("data/heath.xls")
data = excel_data.values
X = data[ : , 1 : ]
Y = data[ : , 0]

breakpoint= int(X.shape[0] * 0.7)
Xtrain = X[ : breakpoint]
Ytrain = Y[ : breakpoint]

Xtest = X[breakpoint : ]
Ytest = Y[breakpoint : ]

model = LinearRegression()
model.fit(Xtrain,Ytrain)
print("score= ", model.score(Xtest,Ytest))
print("predicted results= ", model.predict(Xtest))

# result is so fucking low lol