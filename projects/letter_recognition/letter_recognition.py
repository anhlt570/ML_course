import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Dense, Activation
from keras.models import Sequential
import letter_utils
import os

def read_image(folder):
    imgData = []
    imagelist = glob.glob(folder + '*.jpg')
    ydata = []
    for item in imagelist:
        imgData.append(mpimg.imread(item))
        letter = item.split('\\')[1][0]
        ydata.append(letter_utils.get_letter_id(letter))

    imgData = numpy.array(imgData)
    imgData = imgData/255
    return (imgData, ydata)

def write_result(real,prediction):
    if os.path.exists('output')!= True:
        os.mkdir('output')
    file = open('output/Prediction.csv','w')
    file.write('real,predicted\n')
    for i in range(0, len(prediction)):
        real_letter = str(real[i])
        predicted_letter = str(prediction[i])
        file.write(real_letter+','+predicted_letter+'\n')
    file.close()

# init data
# 0-a, 1-b, 2-c, 3-d, 4-e, 5-f, 6-g
training_data = read_image('training_images/')
verification_data = read_image('verification_images/')
Xtrain = training_data[0]
Ytrain = to_categorical(training_data[1])
Xtest = verification_data[0]
Ytest = to_categorical(verification_data[1])

num_pixels = Xtrain.shape[1]*Xtrain.shape[2]
Xtrain = Xtrain.reshape(Xtrain.shape[0], num_pixels).astype('float32')
Xtest = Xtest.reshape(Xtest.shape[0], num_pixels).astype('float32')
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels,
                kernel_initializer='normal', activation='relu'))
model.add(Dense(Ytest.shape[1], kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(x=Xtrain, y=Ytrain, validation_data=(Xtest, Ytest),
          epochs=10, batch_size=200, verbose=2)

scores = model.evaluate(x=Xtest, y=Ytest, verbose=1)
print('score = ', scores)
prediction_result = [i for i in model.predict(Xtest)]
print(prediction_result)
for i in range(0, len(prediction_result)):
    prediction_result[i] = letter_utils.get_letter_id_onehot(prediction_result[i])

write_result(verification_data[1],prediction_result)
