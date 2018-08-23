import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def get_gender(gender):
    return{
        'Male': 0,
        'Female': 1
        }[gender]

def get_country(country):
    return{
        'France': 0,
        'Spain': 1,
        'Germany': 2
    }[country]

csv_data = pandas.read_csv("data/churn.csv")
data = csv_data.values
data = numpy.delete(data, [0,2], axis=1 )
for item in data:
    item[2] = get_country(item[2])
    item[3] = get_gender(item[3])
data_length = data.shape[0]
X = [data[i][ :-1] for i in range(0,data_length)]
Y = [data[i][-1] for i in range(0,data_length)]

breakpoint= int(data_length* 0.7)
Xtrain = X[ : breakpoint]
Ytrain = Y[ : breakpoint]

Xtest = X[breakpoint : ]
Ytest = Y[breakpoint : ]

#predict by LogisticRegression
logistic_model = LogisticRegression(C=5)
logistic_model.fit(Xtrain, Ytrain)
print("LogisticRegression score = ", logistic_model.score(Xtest,Ytest))
print("LogisticRegression results = ", logistic_model.predict(Xtest))

#predict by DecisionTree
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(Xtrain,Ytrain)
print("DecisionTree score = ", logistic_model.score(Xtest,Ytest))
print("DecisionTree results = ", logistic_model.predict(Xtest))
