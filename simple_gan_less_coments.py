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

class gans:
    #First Structure: Generator structure
    def generator_structure(self,input_generator_shape,output_generator_shape,\
    learning_rate=0.0002,beta=0.5):
        generator_model=keras.models.Sequential()
        generator_model.add(keras.layers.Dense(256,input_dim=input_generator_shape))
        generator_model.add(keras.layers.LeakyReLU(0.2))

        #Add 2 more layer with the same structure
        generator_model.add(keras.layers.Dense(512))
        generator_model.add(keras.layers.LeakyReLU(0.2))
        generator_model.add(keras.layers.Dense(1024))
        generator_model.add(keras.layers.LeakyReLU(0.2))

        #Output layer.
        generator_model.add(keras.layers.Dense(output_generator_shape\
        ,activation='tanh'))

        #Choose the optimizer parameters.
        adam=keras.optimizers.Adam(lr=learning_rate, beta_1=beta)

        #Compitle the model we created.
        generator_model.compile(loss='binary_crossentropy',optimizer=adam)
        return generator_model

    #Create discriminator structure
    def discriminator_model(self, input_discriminator_shape,\
    learning_rate=0.0002,beta=0.5):
        #Feed-fordward model
        discriminator_model=keras.models.Sequential()

        #Input layer and 2nd layer.
        discriminator_model.add(keras.layers.Dense(1024,\
        input_dim=input_discriminator_shape))
        discriminator_model.add(keras.layers.LeakyReLU(0.2))
        discriminator_model.add(keras.layers.Dropout(0.3))

        #Add more layers to the model.
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
    def generative_adversarial_model(self, discriminator ,generator,\
    input_generator_shape):
        #Excludes discriminator from training.
        discriminator.trainable=False

        #Initializes a tensor/model.
        generator_input=keras.layers.Input(shape=(input_generator_shape,))

        #Conect the tensor input with the generator
        generator_output=generator(generator_input)

        #Connect the generated data/generator output with the generator.
        discriminator_output=discriminator(generator_output)

        #Create a model with the goal to map an imput with an output.
        generative_adversarial_network=keras.models.Model(\
        inputs=generator_input, outputs=discriminator_output)

        #Compile the model we just create.
        generative_adversarial_network.compile(loss='binary_crossentropy',\
        optimizer='adam')

        #Return the model
        return generative_adversarial_network

    #Visual analysis of GANs.
    def visual_analysis_generator_data(self,epoch, generator, input_generator_shape,\
    original_data_format,n_samples_to_generate=20, figsize=(12,12) ):
        #Create a random noise.
        random_noise= np.random.normal(loc=0, scale=1,\
        size=[n_samples_to_generate, input_generator_shape])

        #Generate data.
        generated_data=generator.predict(random_noise)

        #Transform the generated data back into the original format to visualize it
        reshape_format=n_samples_to_generate,original_data_format[0],\
        original_data_format[1]
        generated_data=generated_data.reshape(reshape_format)

        #plotting part.
        plt.figure(figsize=figsize)
        dimensions_figure=math.ceil(math.sqrt(n_samples_to_generate))

        for instance_index in range(n_samples_to_generate):
            plt.subplot(dimensions_figure,dimensions_figure,instance_index+1)
            plt.imshow(generated_data[instance_index],interpolation='nearest')
            plt.axis('off')
        #save the figure
        plt.savefig('generative_adversarial_model epoch: '+str(epoch)+'.png')

    #Put together all the functions we have created.
    def generative_adversarial_training_process(self,attributes_training,\
    input_generator_shape, output_generator_shape, original_data_format,\
    epochs=400, batch=128,learning_rate=0.0002, beta=0.5, animation_epoch=30,\
    visualization_samples=56,generation_animation=True,generate_final_data=True,\
    data_to_generate=30,save_models= True):
        #Use the previous functions to generate the generative adversarial model
        #Create the generator using our function
        generator_model = self.generator_structure(input_generator_shape,\
        output_generator_shape, learning_rate=learning_rate, beta=beta)

        #Create the discriminator using our function
        discrimantor_model = self.discriminator_model(output_generator_shape,\
        learning_rate=learning_rate,beta=beta)

        #Create the GAN model using our function
        generative_adversarial_model=self.generative_adversarial_model(\
        discrimantor_model ,generator_model,input_generator_shape)

        #Train the generative adversarial process given
        for epoch in range(1,epochs+1):
            print 'Epoch '+str(epoch)
            for instance_batch in tqdm(range(batch)):
                #Create the input/noise for the generator.
                random_noise=np.random.normal(0,1,[batch,input_generator_shape])

                #Transform the noise into new data.
                generated_data=generator_model.predict(random_noise)

                #Get a batch of random real data from our dataset
                num_intances_training=attributes_training.shape[0]

                #Index to retrieve instance from our dataset
                index_data_to_retrieve=np.random.randint(low=0,\
                high=num_intances_training,size=batch)

                #Retrieve random data from out training set
                real_data_subset=attributes_training[index_data_to_retrieve]

                #Merge the generated data and the real data into a single data set
                discriminator_training_set= np.concatenate(\
                [real_data_subset,generated_data])

                #Create an array with the labels of the generated training set
                labels_training_discriminator=np.zeros(2*batch)

                #the generated instances get the label 0 whereas the real
                #data is label as 0.9
                labels_training_discriminator[:batch]=0.9

                #Train the discriminator to differenciate between generated data
                #and real data.
                discrimantor_model.trainable=True

                discrimantor_model.train_on_batch\
                (discriminator_training_set, labels_training_discriminator)

                #Now we are going to generate data but it will be labbelled as
                #real data.
                random_noise_gans=np.random.normal(0,1,[batch,\
                input_generator_shape])

                #create the labels
                fake_data_labels=np.ones(batch)

                #Once trained deactivative learning process of the discriminator
                discrimantor_model.trainable=False

                #train the generator in the gan model
                generative_adversarial_model.train_on_batch\
                (random_noise_gans,fake_data_labels)

            #Plotting the evolution of the generator
            if generation_animation == True:
                if epoch == 1 or epoch % animation_epoch == 0 or epoch == epochs:
                    self.visual_analysis_generator_data(epoch,\
                    generator_model, input_generator_shape,\
                    original_data_format,visualization_samples,figsize=(12,12))

        #After the training process, generate data that we want to keep for other
        #purposes such as expanding our dataset or Plotting it later.
        if generate_final_data == True:
            #generate noise
            random_noise=np.random.normal(loc=0, scale=1,\
            size=[data_to_generate,input_generator_shape])
            #Generate data.Use the generator we are training to map noise into data
            generated_data=generator_model.predict(random_noise)

            #store the generate data array into a h5py formmat
            with h5py.File('generated_data.h5', 'w') as gan_generation:
                gan_generation.create_dataset("generated_data",\
                data=generated_data)

        #Save the generator, discriminator, and gan model for later usage.
        #We can store the models to use them later if we want to generate more
        #data without any kind of training
        if save_models == True:
            saved_model=generator_model.save('generator_model.h5')
            saved_model=discrimantor_model.save('discrimantor_model.h5')
            saved_model=generative_adversarial_model.save(\
            'generative_adversarial_model.h5')
