import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras import models
from keras import layers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

#filename = 'deltastep.csv'
#filename = 'binary.csv'
#filename = 'normal.csv'
#filename = 'uniform.csv'
raw_data = open(filename, 'rt')
data = np.loadtxt(raw_data, delimiter=",")

#splitting the dataset into the source variables (independant variables) and the target variable (dependant variable)
sourcevars = data[:,:-1] #all columns except the last one
targetvar = data[:,len(data[0])-1] #only the last column

#uniform attack
test, training  = sourcevars[:1219,:], sourcevars[1219:,:]
test_y, training_y  = targetvar[:1219], targetvar[1219:]

#normal distribution attack
#test, training  = sourcevars[:1174,:], sourcevars[1174:,:]
#test_y, training_y  = targetvar[:1174], targetvar[1174:]

#binary attack
#test, training  = sourcevars[:1208,:], sourcevars[1208:,:]
#test_y, training_y  = targetvar[:1208], targetvar[1208:]

#deltestep attack
#test, training  = sourcevars[:1240,:], sourcevars[1240:,:]
#test_y, training_y  = targetvar[:1240], targetvar[1240:]


mean = training.mean(axis=0)
training = training - mean
std = training.std(axis=0)
training = training / std
test = test - mean
test = test / std

print(mean)
print(std)

#note: normalization of test data is always computed using training data.

print("training len:")
print(len(training))
print("test len:")
print(len(test))

print("training y:")
print(len(training_y))
print("test len:")
print(len(test_y))

print("Source")
print(sourcevars)
print("Target")
print(targetvar)


model = models.Sequential()

model.add(layers.Dense(256, activation='relu', input_shape=(training.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
loss='mse',
metrics=['mae'])

#predict output

# checkpoint
# store all better values
filepath="Pu_delta_dis2-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# checkpoint
# store the best only

filepath="Pu_delta_dis2.weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


model.fit(training,
training_y,
epochs=250,
batch_size=128,
validation_data=(test, test_y),
callbacks=callbacks_list, 
verbose=1
)
