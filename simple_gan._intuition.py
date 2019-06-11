#Import Keras tools we use to implements GANs
import tensorflow as tf
from tensorflow import keras

#Import other libraries we are using
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import h5py

#Import dataset
from keras.datasets import mnist

#Code and comment developed by Adrian Salazar Gomez.
#Email:adrianxsalazar@gmail.com

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

#We are going to create two artificial netural network (ANN) structures. One
#learns the distributions of the dataset and replicates the data and the
#other structure differenciates the real data and the created data.
class gans:
    #First Structure: Generator structure
    def generator_structure(self,input_generator_shape,output_generator_shape,\
    learning_rate=0.0002,beta=0.5):

        #The generator model is a simple feedforward artificial neural network.
        generator_model=keras.models.Sequential()

        #Normal Dense layer. the input is the what we call the random noise.
        generator_model.add(keras.layers.Dense(256,input_dim=input_generator_shape))
        #We use the activation unit parametric Rectified Linear Unit: Proposed in
        #'Delving Deep into Rectifiers:Surpassing Human-Level Performance on
        #ImageNet Classification
        generator_model.add(keras.layers.LeakyReLU(0.2))

        #Add 2 more layer with the same structure
        generator_model.add(keras.layers.Dense(512))
        generator_model.add(keras.layers.LeakyReLU(0.2))
        generator_model.add(keras.layers.Dense(1024))
        generator_model.add(keras.layers.LeakyReLU(0.2))

        #Output layer. The output layer has to have the same number of neurons
        #as the attributes that define the real data.
        generator_model.add(keras.layers.Dense(output_generator_shape\
        ,activation='tanh'))

        #Choose the optimizer parameters.Optimizer indicates the way in which
        #the weights that link the neurons are calculated.
        adam=keras.optimizers.Adam(lr=learning_rate, beta_1=beta)

        #Compitle the model we created. Fix final characteristics of the model.
        #The loss, in a simple way, is the criteria we want to minimize.
        generator_model.compile(loss='binary_crossentropy',optimizer=adam)

        #Return the model
        return generator_model

    #Create discriminator structure
    def discriminator_model(self, input_discriminator_shape,\
    learning_rate=0.0002,beta=0.5):
        #Feed-fordward model
        discriminator_model=keras.models.Sequential()

        #Input layer and 2nd layer.The input layer has the same number of neurons
        #as the number of attributes. The number of neurons in the generator
        #output is the same as the number of input neuron in the discriminator.
        discriminator_model.add(keras.layers.Dense(1024,\
        input_dim=input_discriminator_shape))
        discriminator_model.add(keras.layers.LeakyReLU(0.2))

        #Dropout is a regularization technique to avoid overfitting. The technique
        #Inhibits randomly some neurons. For more information about dropout:
        #Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        discriminator_model.add(keras.layers.Dropout(0.3))

        #Add more layers to the model. Mormal connected layers with LReLU and
        #dropout layers
        discriminator_model.add(keras.layers.Dense(512))
        discriminator_model.add(keras.layers.LeakyReLU(0.2))
        discriminator_model.add(keras.layers.Dropout(0.3))
        discriminator_model.add(keras.layers.Dense(256))
        discriminator_model.add(keras.layers.LeakyReLU(0.2))

        #The out layer is binary and differenciates wheter the input is fake
        discriminator_model.add(keras.layers.Dense(1, activation='sigmoid'))

        #Choose the optimizer parameters
        adam=keras.optimizers.Adam(lr=learning_rate, beta_1=beta)

        #compile the model
        discriminator_model.compile(loss='binary_crossentropy',optimizer=adam)

        #return discriminaor model
        return discriminator_model

    #Put together the generator and the discriminator into a single model.
    #We use a previously trained configuration of the discriminatror and we train the
    #generator to produce data that can fool the configuration of the discriminator.
    def generative_adversarial_model(self, discriminator ,generator,\
    input_generator_shape):
        #The command '.trainable=False' excludes a model/layer from training.
        discriminator.trainable=False

        #The input layer initializes a tensor/model. Set up the dimension of the
        #input layer. His input layer has the same dimensions as the
        generator_input=keras.layers.Input(shape=input_generator_shape)

        #Conect the tensor input with the generator
        generator_output=generator(generator_input)

        #Connect the generated data/generator output with the generator.
        discriminator_output=discriminator(generator_output)

        #Create a model with the goal to map an imput with an output. Basically,
        #merging generator and discriminator into a single model. Because the
        #training of the discriminator is locked we train the capacity of our
        #generator to fool the current configurator of the disctiminator.
        generative_adversarial_network=keras.models.Model(\
        inputs=generator_input, outputs=discriminator_output)

        #Compile the model we just create.
        generative_adversarial_network.compile(loss='binary_crossentropy',\
        optimizer='adam')

        #Return the model
        return generative_adversarial_model

    #Visual analysis of GANs. In most of the cases this is only valid if we are
    #using GANs to generate images and we want to visualise the results.
    #We need the epoc
    def visual_analysis_generator_data(self,epoch, generator, n_samples_to_generate,\
    input_generator_shape, original_image_format, figsize=(12,12) ):
        #Create a random noise. The generator transforms the random noise
        #into an adversial samples.
        random_noise=np.random.normal(loc=0, scale=1,\
        size=[n_samples_to_generate,input_generator_shape])
        #Generate data. Use the generator we are training to map noise into data
        generated_data=generator.predict(random_noise)
        #Transform the generated data back into the original format to visualize it
        generated_data=generated_data.reshape(original_image_format)
        #plotting part. Choose the size of the figur.
        plt.figure(figsize=figsize)
        #rows and columns selection is automatic
        dimensions_figure=math.ceil(math.sqrt(n_samples_to_generate))
        for instance_index in range(n_samples_to_generate):
            plt.subplot(dimensions_figure,dimensions_figure,instance_index+1)
            plt.imshow(generated_data[instance_index],interpolation='nearest')
            plt.axis('off')
        #save the figure
        plt.safig('generative_adversarial_model epoch: '+str(epoch)+'.png')

    #Put together all the functions we have created to deliver a GANs structure
    #generative adversarial trining network process
    def generative_adversarial_training_process(self,attributes_training,\
    labels_training, attributes_testing, labels_testing,input_generator_shape,\
    output_generator_shape,epochs=400, batch=100, learning_rate=0.0002, beta=0.5,
    animation_epoch=30, visualization_samples=25, generation_animation=True,
    generate_final_data=True, data_to_generate=30, save_models= True):

        #Use the previous functions to generate the generative adversarial model
        #First, we create the generator and dicriminator. Second, we put both
        #together into a GAN model.
        #Create the generator using our function
        generator_model = self.generator_structure(input_generator_shape,\
        output_generator_shape,learning_rate=learning_rate,beta=beta)
        #Create the discriminator using our function
        discrimantor_model = self.discriminator_model(input_discriminator_shape,\
        learning_rate=learning_rate,beta=beta)
        #Create the GAN model using our function
        generative_adversarial_model=self.generative_adversarial_model(\
        discrimantor_model ,generator_model,input_generator_shape)

        #Process to train the generative adversarial process given a
        #number of epochs or times that we explore the entire dataset and the
        #size of a batch or number of instances we consider before in the
        #training before updating the weights.
        for epoch in range(1,epochs+1):
            #In each epoch we feed our model with a certain number of batches
            #batch is a subset of the dataset
            for instance_batch in range(batch):
                #Create the input/noise for the generator. We have to specify
                #how many instances we want to create and the instance format.
                random_noise=np.random.normal(0,1,[batch,input_generator_shape])

                #Use the structure of the generator to transform the noise into
                #new data.
                generated_data=generator_model.predict(random_noise)

                #Get a batch of random real data from our dataset
                #Number of instances in our dataset to indicate to the random
                #selector the dimension of our training set
                num_intances_training=attributes_training.shape[0]

                #Index to retrieve instance from our dataset
                index_data_to_retrieve=np.random.randint(low=0,\
                high=num_intances_training,size=batch)

                #Retrieve random data from out training set
                real_data_subset=attributes_training[index_data_to_retrieve]

                #Merge the generated data and the real data into a single data set
                discriminator_training_set=np.concatenate(\
                [real_data_subset,generated_data])

                #Create an array with the labels of the generated training set
                labels_training_discriminator=np.zeros(2*batch)

                #the generated instances get the label 0 whereas the real
                #data is label as 0.9
                labels_training_discriminator[:batch]=0.9

                #Train the discriminator to differenciate between generated data
                #and real data. First we unlock the discrimator. Then, we train
                #train the discriminator.
                discrimantor_model.trainable=True
                discrimantor_model.train_on_batch\
                (discriminator_training_set,labels_training_discriminator)

                #Once trained deactivative learning process of the discriminator
                #We do not want the discriminator to learn while the generator
                #is learning to fool the discriminator.
                discrimantor_model.trainable=False

                #Now we are going to generate data but it will be labbelled as
                #real data. Then, discriminator will differenciate this data
                #and the generator will learns to fool the discriminator.
                #First, Generate noise to generate m
                random_noise_gans=np.random.normal(0,1,[batch,input_generator_shape])

                #create the labels
                fake_data_labels=np.ones(batch)

                #train the generator in the gan model
                generative_adversarial_model.train_on_batch\
                (random_noise_gans,fake_data_labels)

            #Plotting the evolution of the generator
            if generation_animation=True:
                if epoch == 1 or epoch % animation_epoch == 0 or epoch == epochs:
                    self.visual_analysis_generator_data(epoch,generator_model,\
                    visualization_samples, input_generator_shape,figsize=(12,12))

        #After the training process, generate data that we can keep for other
        #purposes
        if generate_final_data == True:
            #generate noise
            random_noise=np.random.normal(loc=0, scale=1,\
            size=[data_to_generate,input_generator_shape])
            #Generate data.Use the generator we are training to map noise into data
            generated_data=generator.predict(random_noise)

            #store the generate data array into a h5py formmat
            with h5py.File('generated_data.h5', 'w') as gan_generation:
                gan_generation.create_dataset("generated_data",\
                data=generated_data)

        #Save the generator, discriminator, and gan model for later usage
        if save_models == True:
            saved_model=generator_model.save('generator_model.h5')
            saved_model=discrimantor_model.save('discrimantor_model.h5')
            saved_model=generative_adversarial_model.save(\
            'generative_adversarial_model.h5')


#Download the MNIST dataset
mnist_dataset= keras.datasets.mnist

#Separate training and testing data
(training_attributes, training_labels), (testing_attributes, testing_labels)=\
mnist_dataset.load_data()
