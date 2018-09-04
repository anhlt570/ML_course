import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy
from keras.utils import to_categorical

def read_image(path):
    imgData=[]
    imagelist = glob.glob(path)
    for item in imagelist:
        imgData.append(mpimg.imread(item))
    return imgData

a = read_image('letters/a/*.jpg')
b = read_image('letters/b/*.jpg')
c = read_image('letters/c/*.jpg')
d = read_image('letters/d/*.jpg')
e = read_image('letters/e/*.jpg')
f = read_image('letters/f/*.jpg')
g = read_image('letters/g/*.jpg')
a1 = numpy.array(a)