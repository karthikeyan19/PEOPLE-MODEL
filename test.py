import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.

POS_TEST_DIR = "D:/Python_tutorial/ML/People_Model/test/pos"
NEG_TEST_DIR = "D:/Python_tutorial/ML/People_Model/test/neg"

IMG_SIZE = 256

def create_test_data():
      images = os.listdir(POS_TEST_DIR)
      testing_data = []
      img_num = 1
      for img in os.listdir(POS_TEST_DIR):
        path = os.path.join(POS_TEST_DIR,img)
        img_num += 1
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])    

      images = os.listdir(NEG_TEST_DIR)
      for img in os.listdir(NEG_TEST_DIR):
        path = os.path.join(NEG_TEST_DIR,img)
        img_num += 1
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])    
      
      return testing_data


test_data = create_test_data()
print(len(test_data))    
shuffle(test_data)
np.save('test_data.npy', test_data)

    


