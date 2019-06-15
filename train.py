from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K


#Model with 5 CNNs and 4 FCLs.
def create_nvidia_model1():
    #with tf.device('/cpu:0'):
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, input_shape=(row, col, ch)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Flatten())

	# Added a Dropout layer to help reduce overfitting. 
    model.add(Dropout(0.5))

    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    #model=multi_gpu_model(model,gpus=1)
    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model

def my_train_generator():
    num_iters = X_train.shape[0] / batch_size
    num_iters = int(num_iters)
    while 1:
        for i in range(num_iters):
            #idx = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
            idx = train_idx_shf[i*batch_size:(i+1)*batch_size]
            tmp = X_train[idx].astype('float32')
            tmp = tmp - X_train_mean
            tmp = tmp / 255.0
            yield tmp, y_train[idx]

def my_test_generator():
    num_iters = X_test.shape[0] / batch_size
    num_iters = int(num_iters)
    while 1:
        for i in range(num_iters):
            tmp = X_test[i*batch_size:(i+1)*batch_size].astype('float32')
            tmp = tmp - X_train_mean
            tmp = tmp / 255.0
            yield tmp, y_test[i*batch_size:(i+1)*batch_size]

            
if __name__ == "__main__":

    print(device_lib.list_local_devices())
    K.tensorflow_backend._get_available_gpus()
    row, col, ch =  192,256,3
    num_epoch,batch_size= 25,32
    data_path = "G:/Project/nvidia4/data/"
    model_name="nvidia4"                   
    
    print ("Loading training data...")
    print ("Data path: " + data_path + "X_train_preprocess.npy")

    X_train = np.load(data_path + "X_train_preprocess1.npy")
    y_train = np.load(data_path + "y_train_preprocess1.npy")
    
    X_test = np.load(data_path + "X_train_preprocess3.npy")
    y_test = np.load(data_path + "y_train_preprocess3.npy")
               
    print ("X_train shape:" + str(X_train.shape))
    print ("X_test shape:" + str(X_test.shape))
    print ("y_train shape:" + str(y_train.shape))
    print ("y_test shape:" + str(y_test.shape))
    
    np.random.seed(1235)
    train_idx_shf = np.random.permutation(X_train.shape[0])
    
    print ("Computing training set mean...")
    X_train_mean = np.mean(X_train, axis=0, keepdims=True)
    
    print ("Saving training set mean...")
    np.save("data/X_train_"+model_name+"_mean.npy", X_train_mean)
    
    print ("Creating model...")
    model = create_nvidia_model1()
    print (model.summary())

    # checkpoint
    filepath = data_path + "models/weights_" + model_name + "-{epoch:02d}-{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,period=1)
    callbacks_list = [checkpoint]
    
    iters_train = X_train.shape[0]
    iters_train = iters_train - iters_train % batch_size
    iters_test = X_test.shape[0]
    iters_test = iters_test - iters_test % batch_size
    
    model.fit_generator(my_train_generator(),
        nb_epoch=num_epoch,
        samples_per_epoch=iters_train,
        validation_data=my_test_generator(),
        nb_val_samples=iters_test,
        callbacks=callbacks_list,
        nb_worker=1,
        verbose=1
    )
