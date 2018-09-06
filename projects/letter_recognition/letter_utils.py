import numpy


def get_letter_id(letter):
    return {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
        'f': 5,
        'g': 6,
    }[letter]


def get_letter(letter_id):
    return chr(ord('a') + letter_id)

def get_letter_id_onehot(code):
    array = numpy.array(code)
    imax=0
    for i in range(0, array.shape[0]):
        if array[i] >array[imax]:
            imax = i
    return imax
