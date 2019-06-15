"""
Preprocessing: Data Augmentation added.
"""
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
from skimage.exposure import rescale_intensity
from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
import glob
    
def preprocess_train(path, num_channels):
    df = pd.read_csv(path)
    num_rows = df.shape[0]
    
    X = np.zeros((num_rows, row, col, num_channels), dtype=np.uint8)
    for i in range( num_rows):
        if i % 1000 == 0:
            print ("Processed " + str(i) + " images...")
        
        path = df['fullpath'].iloc[i]
        img = load_img(data_path + path, target_size=(row, col))
        img = img_to_array(img)
        img = rgb_to_hsv(img)
        img = np.array(img, dtype=np.uint8)
        X[i, :, :, :] = img
    return X, np.array(df["angle"])

def preprocess_test(num_channels):
    num_rows = len(filenames)
    
    X = np.zeros((num_rows, row, col, num_channels), dtype=np.uint8)
    for i in range( num_rows):
        if i % 1000 == 0:
            print ("Processed " + str(i) + " images...")
        
        path = filenames[i]
        img = load_img(path, target_size=(row, col))
        img = img_to_array(img)
        img = rgb_to_hsv(img)
        img = np.array(img, dtype=np.uint8)
        X[i, :, :, :] = img

    return X    
if __name__ == "__main__":
    data_path = "G:/Project/nvidia4/data/"
    row, col, ch =  192,256,3
   
    print ("Pre-processing train data set1...")
    i=1
    X_train, y_train = preprocess_train("data/train" + str(i) + ".txt", ch)
    np.save("{}/X_train_preprocess{}".format(data_path, i), X_train)
    np.save("{}/y_train_preprocess{}".format(data_path, i), y_train)

    print ("Pre-processing validation data set3...")
    i=3
    X_train, y_train = preprocess_train("data/train" + str(i) + ".txt", ch)
    np.save("{}/X_train_preprocess{}".format(data_path, i), X_train)
    np.save("{}/y_train_preprocess{}".format(data_path, i), y_train)

    print ("Pre-processing test data set...")
    filenames = glob.glob("{}/test/center/*.jpg".format(data_path))
    filenames = sorted(filenames)
    X_test = preprocess_test(ch)
    np.save("{}/X_test_preprocess".format(data_path), X_test)