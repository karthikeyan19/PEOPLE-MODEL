import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.

POS_TRAIN_DIR = "D:/Python_tutorial/ML/People_Model/train/pos"
NEG_TRAIN_DIR = "D:/Python_tutorial/ML/People_Model/train/neg"
IMG_SIZE = 256
pos_images = os.listdir(POS_TRAIN_DIR)
neg_images = os.listdir(NEG_TRAIN_DIR)
data = []

def label_img(img,word_label):
    #print(word_label)
    # conversion to one-hot array [bw,captain,pool,hulk,iron,spider,thor]
    if word_label == 'people':
        print(word_label)
        return [1,0]
    elif word_label == 'not':
        print(word_label)
        return [0,1]
    
def create_train_data():
    
    training_data = []
    
    for i in range(len(pos_images)):
        img = pos_images[i]
        label = label_img(img,'people')
        path = os.path.join(POS_TRAIN_DIR,img)
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    for i in range(len(neg_images)):
        img = neg_images[i]
        label = label_img(img,'not')
        path = os.path.join(NEG_TRAIN_DIR,img)
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
        
    return training_data

trained_data = []
trained_data = create_train_data()
print(len(trained_data))    
shuffle(trained_data)
np.save('train_data.npy', trained_data)

    


