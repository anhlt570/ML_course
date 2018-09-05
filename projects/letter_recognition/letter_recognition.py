import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Dense, Activation
from keras.models import Sequential

def get_text(text_id):
    return {
        'a':0
    }[text_id]

def read_image(folder):
    imgData=[]
    imagelist = glob.glob(folder + '*.jpg')
    for item in imagelist:
        imgData.append(mpimg.imread(item))
    imgData = numpy.array(imgData)
    imgData = imgData/255
    return imgData

# init data
# 0-a, 1-b, 2-c, 3-d, 4-e, 5-f, 6-g
Xtrain = read_image('training_images/')
Xtest = read_image('verification_images/')
Ytrain = numpy.array([0] * 40 + [1] * 40 + [2] * 40 + [3] * 40 + [4] * 40 + [5] * 40 + [6] * 40)
Ytrain = to_categorical(Ytrain)
Ytest = numpy.array([0] * 15 + [1] * 15 + [2] * 15 + [3] * 15 + [4] * 15 + [5] * 15 + [6] * 15)
Ytest = to_categorical(Ytest)
num_pixels = 60*80*3
Xtrain = Xtrain.reshape(Xtrain.shape[0], num_pixels).astype('float32')
Xtest = Xtest.reshape(Xtest.shape[0], num_pixels).astype('float32')
model = Sequential()
model.add(Dense(num_pixels,input_dim= num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(Ytest.shape[1],kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x=Xtrain,y= Ytrain, validation_data=(Xtest,Ytest), epochs=5, batch_size=200, verbose=2,shuffle=True)

scores = model.evaluate(x = Xtest,y= Ytest, verbose=0)
print('score = ',scores)
print('prediction= ',model.predict(Xtest))
