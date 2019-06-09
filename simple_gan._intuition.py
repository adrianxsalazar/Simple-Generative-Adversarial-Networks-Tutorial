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

    #The generator model is a simple feedforward artificial neural network.
    generator_model=keras.models.Sequential()

    #Normal Dense layer. the input is the what we call the random noise.
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

    #Output layer. The output layer has to have the same number of neurons
    #as the attributes that define the real data.
    generator_model.add(keras.layers.Dense(output_shape, activation='tanh'))

    #Compitle the model we have created. Fix final characteristics of the model.
    #Optimizer indicates the way in which the weights that link the neurons
    #are calculated.The loss, in a simple way, is the in the criteria that we
    #try to minimize.
    generator_model.compile(loss='binary_crossentropy',optimizer='adam')

    #Return the model
    return generator_model

#Create discriminator structure
def discriminator_model(input_dimension):
    #Feed-fordward model
    discriminator_model=keras.models.Sequential()

    #Input layer and 2nd layer. The input layer has the same number of neurons
    #as the number of attributes.
    discriminator_model.add(keras.layers.Dense(1024, input_dim=input_dimension))
    discriminator_model.add(keras.layers.LeakyReLU(0.2))

    #Dropout is a regularization technique to avoid overfitting. The technique
    #Inhibits randomly some neurons. For more information about dropout:
    #Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    discriminator_model.add(keras.layers.Dropout(0.3))

    #Add more layers to the model. Mormal connected layers with LReLU and drop
    discriminator_model.add(keras.layers.Dense(1024, input_dim=input_dimension))
    discriminator_model.add(keras.layers.LeakyReLU(0.2))
    discriminator_model.add(keras.layers.Dropout(0.3))
    discriminator_model.add(keras.layers.Dense(1024, input_dim=input_dimension))
    discriminator_model.add(keras.layers.LeakyReLU(0.2))

    #The out layer is binary and differenciates wheter the input is fake
    discriminator_model.add(keras.layers.Dense(784, activation='tanh'))

    #compile the model
    discriminator_model.compile(loss='binary_crossentropy',optimizer='adam')

    return discriminator_model

#Put together the generator and the discriminator
def generative_adversarial_model(discriminator, generator):
    #The command '.trainable=False' excludes a model/layer from training
    discriminator.trainable=False

    #The input layer initializes a tensor/model
    keras.layers.Input(shape=)

    #Create



    return generative_adversarial_model


#Visual analysis
#Put together all the functions we have created to deliver a GANs structure
