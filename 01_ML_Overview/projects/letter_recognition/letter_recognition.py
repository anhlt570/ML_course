import glob
import numpy
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten,MaxPooling2D,Conv2D
from keras.models import Sequential
import letter_utils
import os
from sklearn.neural_network import MLPClassifier
import datetime
import model_utils
import data_utils


def model_by_MLP(Xtrain, Ytrain):
    model = MLPClassifier()
    model.fit(Xtrain, Ytrain)
    return model


def model_by_keras(Xtrain, Ytrain, num_pixels):
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels,
                    kernel_initializer='normal', activation='relu'))
    model.add(
        Dense(Ytest.shape[1], kernel_initializer='normal', activation='softmax'))
    # model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 1),activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(Ytrain.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    model.fit(x=Xtrain, y=Ytrain,  epochs=100, batch_size=200)
    return model


# init data
# 0-a, 1-b, 2-c, 3-d, 4-e, 5-f, 6-g
training_data = data_utils.ReadImagesFromFolder('training_images/')
verification_data = data_utils.ReadImagesFromFolder('verification_images/')
Xtrain = training_data[0]
Ytrain = to_categorical(training_data[1])
Xtest = verification_data[0]
Ytest = to_categorical(verification_data[1])

num_pixels = Xtrain.shape[1]*Xtrain.shape[2]
Xtrain = Xtrain.reshape(Xtrain.shape[0], num_pixels)
Xtest = Xtest.reshape(Xtest.shape[0], num_pixels)

model = model_by_keras(Xtrain, Ytrain, num_pixels) 
print('score with test data= ',model.evaluate(Xtest,Ytest))
model_utils.WriteKerasModelToFile(model)
