#Import Keras tools we use to implements GANs
import tensorflow as tf
from tensorflow import keras

#Import other libraries we are using
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import h5py
from tqdm import tqdm

from simple_gans import gans


#Import dataset
from keras.datasets import mnist

#Get the dataset from Keras.
mnist_dataset= keras.datasets.mnist

#Separate training and testing data. Load the mnist dataset into an array
(attributes_training, labels_training), (attributes_testing, labels_testing)=\
mnist_dataset.load_data()

#Transform the input values into a 1 , -1 scale.
attributes_training=(attributes_training.astype(np.float32)-127.5)/127.5

#Reshape the instances of our data set in a way that we can use it in normal
#neural networks. Our original dataset were images of 28x28 resulting in 784
#pixel
attributes=attributes_training.reshape(60000,784)

#We need to store the original dimension of our data to then after the
#training process of the generator map the generator output to the original
#format
original_image_format=28,28

#get the class from the file 'simple_gans.py'
generative_adversarial_network=gans()

#Implement the gan model that we have created.
generative_adversarial_network.\
generative_adversarial_training_process(attributes,100, 784,\
original_image_format)
