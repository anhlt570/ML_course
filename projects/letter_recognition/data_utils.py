import glob
import letter_utils
import matplotlib.image as mpimg
from PIL import Image
import numpy

def ReadImagesFromFolder(folder):
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

def ReadImageFromFile(path):
    image_data =Image.open(path).convert('L').resize((32,32))
    imgData = numpy.array(image_data)
    imgData = imgData/255
    return imgData

def ReadImageFromByteData(byte_data):
    image_data = Image.frombytes('L',(32,32),byte_data)
    print('data from byte = ',image_data)
    imgData = numpy.array(image_data)
    imgData = imgData/255
    return imgData