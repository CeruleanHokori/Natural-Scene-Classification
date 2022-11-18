import os
import random
import cv2
import matplotlib as plt
from sklearn.model_selection import train_test_split
import numpy as np
W,H,C = 150,150,3 #Width Height and Channels

train_path,test_path ="data/seg_train/seg_train","data/seg_test/seg_test"

#Classes imgs, we start by training over 3 classes
buildings_train = [train_path+r"/buildings/"+i for i in os.listdir(train_path+r"/buildings")]
forest_train = [train_path+r"/forest/"+i for i in os.listdir(train_path+r"/forest")]
sea_train = [train_path+r"/sea/"+i for i in os.listdir(train_path+r"/sea")]

#All classes in one list
imgs_train = buildings_train+forest_train+sea_train
random.shuffle(imgs_train) #! We don't want our system to learn the order

###Same for testing
buildings_test = [test_path+r"/buildings/"+i for i in os.listdir(test_path+r"/buildings")]
forest_test = [test_path+r"/forest/"+i for i in os.listdir(test_path+r"/forest")]
sea_test = [test_path+r"/sea/"+i for i in os.listdir(test_path+r"/sea")]
imgs_test = buildings_test+forest_test+sea_test
random.shuffle(imgs_test)

#label encoder
def encoder(class_name):
    if class_name == "buildings":
        return np.array([1,0,0])
    elif class_name == "forest":
        return np.array([0,1,0])
    elif class_name == "sea":
        return np.array([0,0,1])

#Extracts samples and labels from a list of images
def samples_labels(img_list):
    X,y=[],[]
    for filename in img_list:
        img = cv2.imread(filename)
        X.append(cv2.resize(img,dsize=(W,H), interpolation=cv2.INTER_CUBIC)) #In our case resizing is not necessary
        if "buildings" in filename:
            y.append(encoder("buildings"))
        elif "forest" in filename:
            y.append(encoder("forest"))
        elif "sea" in filename:
            y.append(encoder("sea"))
    return X,y


#Validation set splitter
def val_split(X,y,ratio=0.2):
    return train_test_split(X,y,test_size=ratio,random_state=42)






