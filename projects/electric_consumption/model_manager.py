from keras.models import load_model
import os

def SaveElectricConsumptionModel(model):
    if not os.path.exists('models'):
        os.mkdir('models')
    model.save('models/electric_consumption.h5')


def LoadElectricConsumptionModel():
    model_path = 'models/electric_consumption.h5'
    if not os.path.exists(model_path):
        return -1
    return load_model(model_path)