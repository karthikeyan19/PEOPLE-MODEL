import tflearn
import numpy as np
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os                  # dealing with directories
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn.data_utils as du
from sklearn.model_selection import train_test_split

train_data = np.load('train_data.npy')


IMG_SIZE = 256
LR = 1e-3

MODEL_NAME = 'people-{}-{}.model'.format(LR, '1_convnet') # just so we remember which saved model is which, sizes must match
def bulid_model():
        
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 8, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
     
    convnet = conv_2d(convnet, 16, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    
    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    
    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    
    convnet = conv_2d(convnet, 128, 3, activation='relu')
    #convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 256, 3, activation='relu')
    #convnet = max_pool_2d(convnet, 3)
    

    convnet = tflearn.global_avg_pool(convnet)
    convnet = fully_connected(convnet, 512, activation='relu')
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, 2, activation='softmax')
    
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log',checkpoint_path='model_marvel',max_checkpoints=10)
  
    return model

def train(model):
    train = train_data

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    Y = [i[1] for i in train]
    X , test_x , Y , test_y = train_test_split(X, Y, test_size=0.33)
    model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=500,batch_size=10, show_metric=True, run_id=MODEL_NAME)
    return model

def load_model(model):
    model.load('./save/marvel_model')
    
def save_model(model):
    return model.save('./save/marvel_model')

def test_data(model):
    test_data = np.load('test_data.npy')
    fig=plt.figure()

    for num,data in enumerate(test_data[12:24]):
        # bw: [1,0,0,0,0,0]
        # captain: [0,1,0,0,0,0]
        # pool: [0,0,1,0,0,0]
        # hulk: [0,0,0,1,0,0]
        # iron: [0,0,0,0,1,0]
        # spider: [0,0,0,0,0,1]
        
        img_num = data[1]
        img_data = data[0]
        
        y = fig.add_subplot(3,4,num+1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
        
        
        #predict
        model_out = model.predict([data])
        model_out = model_out[0]
        
        switcher = {
        1: 'People',
        2: 'Notppl',
        }
        # Get the function from switcher dictionary
        str_label = switcher.get(np.argmax(model_out)+1, lambda: "Invalid")    
        y.imshow(orig)
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()

#buliding model
model = bulid_model()

#training model
#model = train(model)

#saving Model
#save_model(model)

#loading from saved model
load_model(model)

#testing data
test_data(model)


