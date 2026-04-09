import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dense,Flatten
from tensorflow.keras.optimizers import Adam



def getName(filePath):
    return filePath.split('\\')[-1]

def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    # print(data['Center'][0])
    # print(getName(data['Center'][0]))
    
    data['Center'] = data['Center'].apply(getName)
    print(data.head())
    # print(data.columns.tolist())
    print("Total images imported: " + str(data.shape[0]))
    return data

def balanceData(data,display=True):
    nBin = 31
    samplesPerBin = 1000
    hist, bins = np.histogram(data['Steering'], nBin)
    # print(bins)
    if display:
        Center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(Center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()

    removeIndexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print("Removed Images: " + str(len(removeIndexList)))
    data.drop(data.index[removeIndexList], inplace=True)
    print("Remaining Images: " + str(len(data)))

    if display:
        hist, _ = np.histogram(data['Steering'], nBin)
    
        plt.bar(Center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()

    return data 


def loadData(path, data):
    imagesPath = []
    steering = []
    for i in range(len(data)):
        indexedData = data.iloc[i]
        # print(indexedData)
        imagesPath.append(os.path.join(path, 'IMG', indexedData['Center']))
        # print(os.path.join(path, 'IMG', indexedData['Center']))
        steering.append(float(indexedData['Steering']))
    imagesPath = np.array(imagesPath)
    steering = np.array(steering)
    return imagesPath, steering 



def augmentImage(imgPath,steering):
    img = mpimg.imread(imgPath)
    # PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)})
        img = pan.augment_image(img)
    #ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    # #brightness
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4,1.2))
        img = brightness.augment_image(img)

    #flipping   
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering = -steering


    return img, steering

    # if np.random.rand() < 0.5:
    #     img = iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)}).augment_image(img)
    # if np.random.rand() < 0.5:
    #     img = iaa.Affine(scale=(0.8,1.2)).augment_image(img)
    # if np.random.rand() < 0.5:
    #     img = iaa.Affine(rotate=(-15,15)).augment_image(img)
    # if np.random.rand() < 0.5:
    #     img = iaa.Add((-10,10)).augment_image(img)
    # if np.random.rand() < 0.5:
    #     img = iaa.Multiply((0.8,1.2)).augment_image(img)
    # if np.random.rand() < 0.5:
    #     img = iaa.GaussianBlur(sigma=(0,1)).augment_image(img)

    # return img, steering




# imRe,st = augmentImage('selfDrivingSimulation\\test.jpg',0)
# plt.imshow(imRe)
# plt.show()


def preProcessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img/255
    return img


# img = mpimg.imread('selfDrivingSimulation\\test.jpg')  # read the image first
# imRe = preProcessing(img)  
# plt.imshow(imRe)
# plt.show()

def batchGen(imagesPath, steeringList, batchSize,trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath)-1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch), np.asarray(steeringBatch)) 



def createModel():
    model = Sequential()
    model.add(Convolution2D(24, (5,5), (2,2), input_shape=(66,200,3), activation='elu'))
    model.add(Convolution2D(36, (5,5), (2,2), activation='elu'))
    model.add(Convolution2D(48, (5,5), (2,2), activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model