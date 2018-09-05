from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical

(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
num_pixels = Xtrain.shape[1] * Xtrain.shape[2]
Xtrain = Xtrain.reshape(Xtrain.shape[0], num_pixels).astype('float32')
Xtest = Xtest.reshape(Xtest.shape[0], num_pixels).astype('float32')
Xtrain = Xtrain / 255
Xtest = Xtest / 25
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)
num_classes = ytest.shape[1]
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=10,
batch_size=200, verbose=2)
scores = model.evaluate(Xtest, ytest, verbose=0)
print("score=", scores[1])
f = open('mnist.json', 'w')
f.write(model.to_json())
f.close()
model.save_weights('mnist.h5')