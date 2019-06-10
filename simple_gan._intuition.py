#Import Keras tools we use to implements GANs
import tensorflow as tf
from tensorflow import keras

#Import other libraries we are using
import numpy as np
from IPython.core.debugger import Tracer
import math
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
def generator_structure(input_generator_shape, output_generator_shape,\
learning_rate=0.0002,beta=0.5):

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
    generator_model.add(keras.layers.Dense(output_generator_shape\
    ,activation='tanh'))

    #Choose the optimizer parameters
    adam=keras.optimizers.Adam(lr=learning_rate, beta_1=beta)

    #Compitle the model we have created. Fix final characteristics of the model.
    #Optimizer indicates the way in which the weights that link the neurons
    #are calculated.The loss, in a simple way, is the in the criteria that we
    #try to minimize.
    generator_model.compile(loss='binary_crossentropy',optimizer=adam)

    #Return the model
    return generator_model

#Create discriminator structure
def discriminator_model(input_dimension, learning_rate=0.0002,beta=0.5):
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

    #Choose the optimizer parameters
    adam=keras.optimizers.Adam(lr=learning_rate, beta_1=beta)

    #compile the model
    discriminator_model.compile(loss='binary_crossentropy',optimizer=adam)

    return discriminator_model

#Put together the generator and the discriminator into a single model
def generative_adversarial_model(discriminator,generator,input_generator_shape):
    #The command '.trainable=False' excludes a model/layer from training
    discriminator.trainable=False

    #The input layer initializes a tensor/model
    generator_input=keras.layers.Input(shape=input_generator_shape)

    #We input a noise
    generator_output=generator(generator_input)

    #Model output
    discriminator_output=discriminator(generator_output)

    #Create a model with the goal to map an imput with an output. Basically,
    #merging generator and discriminator into a single model
    generative_adversarial_network=keras.models.Models(\
    inputs=generator_input, outputs=discriminator_output)

    #Compile the model we just create
    generative_adversarial_network.compile(loss='binary_crossentropy',\
    optimizer='adam')

    return generative_adversarial_model


#Visual analysis of GANs: Code taken from
def visual_analysis_generator_data(epoch,generator,n_samples,\
input_generator_shape, figsize=(12,12) ):
    #Create a random noise. The generator will transform the random noise
    #into an adversial sample.
    random_noise=np.random.normal(0,1,size=[n_samples,input_generator_shape])
    #Generate data
    generated_data=generator.predict(random_noise)
    #
    generated_data=generated_data.reshape()
    #
    plt.figure(figsize=figsize)
    #
    dimensions_figure=math.ceil(math.sqrt(n_samples))
    #
    for instance_index in range(n_samples):
        plt.subplot(dimensions_figure,dimensions_figure,instance_index+1)
        plt.imshow(generated_data[instance_index],interpolation='nearest')
        plt.axis('off')
    plt.safig('generative_adversarial_model epoch: '+str(epoch)+'.png')

#Put together all the functions we have created to deliver a GANs structure

#generative adversarial trining network process
def generative_adversarial_training_process(epochs=1, batch=100,
attributes_training, labels_training, attributes_testing, labels_testing,
input_generator_shape,output_generator_shape,learning_rate=0.0002, beta=0.5,
generation_animation=True):

    #Use the previous functions to generate the generative adverial model
    #First we create the generator and dicriminator. Second, we put both
    #together into a GAN model.
    generator_model=
    discrimantor_model=
    generative_adversarial_model=

    #
    for epoch in range(1,epochs+1):
        for instance_batch in range(batch):

            #Create the input for the generator. The input
            #
            random_noise=np.random.normal(0,1,[batch,input_generator_shape])

            #Use the generator
            generated_data=generator_model.predict(random_noise)

            #Get a batch of random real data from our dataset
            num_intances_training=attributes_training.shape[0]

            index_data_to_retrieve=np.random.randint(low=0,\
            high=num_intances_training,size=batch)

            real_data_subset=attributes_training[index_data_to_retrieve]

            #Merge the generated data and the real data into a single data set
            discriminator_training_set=np.concatenate(\
            [real_data_subset,generated_data])

            #Label the discriminator training set
            labels_training_discriminator=np.zeros(2batch)

            #the generated instances get the lable 0.9
            labels_training_discriminator[:batch]=0.9

            #Train the discriminator to differenciate between generated data
            #and real data. This initializes the GAN process
            discrimantor_model.trainable=True
            discrimantor_model.train_on_batch\
            (discriminator_training_set,labels_training_discriminator)

            #Generate a data
            random_noise_gans=np.random.normal(0,1,[batch,input_generator_shape])
            fake_data_labels=np.ones(batch)

            #deactivative learning process of the discriminator
            discrimantor_model.trainable=False

            #
            generative_adversarial_model.train_on_batch\
            (random_noise_gans,fake_data_labels)

        if generation_animation=True:
            if epoch == 1 or epoch:
                visual_analysis_generator_data(epoch,generator,n_samples,\
                input_generator_shape, figsize=(12,12) )
