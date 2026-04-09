print("setting up the model")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from utlis import *
from sklearn.model_selection import train_test_split
#step1
path = 'selfDrivingSimulation\\myData'

data = importDataInfo(path)
#step 2
balanceData(data,display=False)


# step 3

# loadData(path, data)
imagesPath, steering = loadData(path, data)

# print(imagesPath[0])
# print(steering[0])

# step 4
xTrain,xVal,yTrain,yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print("Training Samples: " + str(len(xTrain)))
print("Validation Samples: " + str(len(xVal)))


# step 5

#step 6


#step 7

#step 8
model = createModel()
model.summary()

#step 9
history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=10, validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)



#step 10
model.save('model.h5')
print("model saved")

plt.plot(history.history['loss'] )
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation loss'])

plt.title('Loss')
plt.xlabel('Epoch')
plt.show()





