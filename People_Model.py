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
LR = 1e-4

MODEL_NAME = 'people-{}-{}.model'.format(LR, '1_convnet') # just so we remember which saved model is which, sizes must match
def bulid_model():
        
    tf.reset_default_graph()

    # Building Residual Network
    net = tflearn.input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3])
    net = tflearn.conv_2d(net, 128, 3, activation='relu', bias=False)
    # Residual blocks
    net = tflearn.residual_bottleneck(net, 3, 16, 128)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.residual_bottleneck(net, 1, 32, 256, downsample=True)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')

    net = tflearn.residual_bottleneck(net, 2, 32, 256)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')

    net = tflearn.residual_bottleneck(net, 1, 64, 512, downsample=True)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')

    net = tflearn.residual_bottleneck(net, 2, 64, 512)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    # Regression
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=LR)

    model = tflearn.DNN(net, tensorboard_dir='log',checkpoint_path='model_marvel',max_checkpoints=10)
  
    return model

def train(model):
    train = train_data

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    Y = [i[1] for i in train]
    X , test_x , Y , test_y = train_test_split(X, Y, test_size=0.5)
    model.fit(X, Y, n_epoch=5, validation_set=(test_x, test_y), 
    show_metric=True ,batch_size=1,  run_id=MODEL_NAME)

    return model

def load_model(model):
    model.load('./save/people_model')
    
def save_model(model):
    return model.save('./save/people_model')

def test_data(model):
    test_data = np.load('test_data.npy')
    fig=plt.figure()

    for num,data in enumerate(test_data[1:12]):
        # people: [1,0]
        # no: [0,1] 
        img_num = data[1]
        img_data = data[0]
        
        y = fig.add_subplot(3,4,num+1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
        
        #predict
        model_out = model.predict([data])[0]
        switcher = {
        1: 'People',
        2: 'Not',
        }
        # Get the function from switcher dictionary
        str_label = switcher.get(np.argmax(model_out)+1, lambda: "Invalid")    
        y.imshow(orig)
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()
    for num,data in enumerate(test_data):
        # people: [1,0]
        # no: [0,1] 
        img_num = data[1]
        img_data = data[0]
        
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
        
        #predict
        model_out = model.predict([data])[0]
        switcher = {
        1: 'People',
        2: 'Not',
        }
        # Get the function from switcher dictionary
        str_label = switcher.get(np.argmax(model_out)+1, lambda: "Invalid")    
        print(str_label)
        
#buliding model
model = bulid_model()

#loading from saved model
load_model(model)

#training model
model = train(model)

#saving Model
save_model(model)

#loading from saved model
#load_model(model)

#testing data
test_data(model)


