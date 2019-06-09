#Import Keras tools we use to implements GANs
import tensorflow as tf
from tensorflow import keras

#Import other libraries we are using
import numpy as np
from IPython.core.debugger import Tracer
import matplotlib.pyplot as plt
import os

#Import dataset
from keras.datasets import mnist

#Introduction: The purpose of this script is to provide a practical exercise
#to understand generative adversial networks and the implementation code.
#The script includes multiple comments that explain the process implemented
#on the lines below the comment. In the script, first we create fucntions that
#represent the basic elements of a GAN structure and then we put all the
#elements together to create a GAN model.

#This code was inpired by the code presented in:
#https://medium.com/@mattiaspinelli/simple-generative-adversarial-network-gans\
#-with-keras-1fe578e44a87
#https://www.tensorflow.org/beta/tutorials/generative/dcgan
#https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using\
#-keras-ce1c05cfdfd3

#And the theory presented in the tutorial:
#NIPS 2016 Tutorial: Generative Adversarial Networks.


#Download the MNIST dataset
mnist_dataset= keras.datasets.mnist

#Separate training and testing data
(training_attributes, training_labels), (testing_attributes, testing_labels)=\
mnist_dataset.load_data()

#We are going to create two artificial netural network (ANN) structures. One
#learns the distributions of the dataset and replicates the data and the
#other structure differenciates the real data and the created data.

#First Structure: Generator structure
def generator_structure(input_generator_shape, output__discriminator_shape):

    #The
    generator_model=keras.models.Sequential()

    #Normal Dense layer
    generator_model.add(keras.layers.Dense(256, input_dim=input_shape))
    #We use the activation unit parametric Rectified Linear Unit: Proposed in
    #'Delving Deep into Rectifiers:Surpassing Human-Level Performance on
    #ImageNet Classification
    generator_model.add(keras.layers.LeakyReLU(0.2))

    #Add 2 more layer with the same structure
    generator_model.add(keras.layers.Dense(256))
    generator_model.add(keras.layers.LeakyReLU(0.2))
    generator_model.add(keras.layers.Dense(1024))
    generator_model.add(keras.layers.LeakyReLU(0.2))

    #Output layer
    generator_model.add(keras.layers.Dense(output_shape, activation='tanh'))

    #Compitle the model we have created. Fix final characteristics of the model.
    #Optimizer indicates the way in which the weights that link the neurons
    #are calculated.The loss, in a simple way, is the in the criteria that we
    #try to minimize.
    generator_model.compile(loss='binary_crossentropy',optimizer='adam')

    #Return the model
    return generator_model

#Put together all the functions we have created to deliver a GANs structure


#Create discriminator structure
def discriminator_model():
    #
    discriminator_model=keras.models.Sequential()
    #
    discriminator_model.add()
    #
    discriminator_model.add()
    #

    #

    #
    return discriminator_model
