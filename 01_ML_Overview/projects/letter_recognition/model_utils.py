import os
import pickle
import datetime
import glob
import keras
from keras.models import load_model
import letter_utils
import data_utils

def WriteKerasModelToFile(model):
    if not os.path.exists('models'):
        os.mkdir('models')

    current_models = glob.glob('models/'+'keras*')
    for item in current_models:
        os.remove(item)
    
    path = 'models/keras_' + datetime.datetime.now().strftime('%Y%m%d_%H%M')+'.h5'
    model.save(path)


def ReadKerasModelFromFile(path):
    if not os.path.exists(path):
        return -1
    return load_model(path)

def PredictByKeras(image_data):
    model_path = glob.glob('models/keras*.h5')[0]
    model = ReadKerasModelFromFile(model_path)
    image_data = image_data.reshape(1,image_data.shape[0]*image_data.shape[1]).astype('float32')
    onehot_result = model.predict(image_data)[0]
    result_id = letter_utils.get_letter_id_onehot(onehot_result)
    return letter_utils.get_letter(result_id)

# print (PredictByKeras(data_utils.read_image('images/recognize/temp_image.jpg')))