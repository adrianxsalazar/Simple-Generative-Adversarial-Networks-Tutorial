#Download the MNIST dataset
from simple_gans import gans

#Import dataset
from keras.datasets import mnist

#
mnist_dataset= keras.datasets.mnist

#Separate training and testing data. Load the mnist dataset into an array
(attributes_training, labels_training), (attributes_testing, labels_testing)=\
mnist_dataset.load_data()

#Transform the input into a 1 , -1 scale
attributes_training=(attributes_training.astype(np.float32)-127.5)/127.5

#
attributes=attributes_training.reshape(60000,784)

#
original_image_format=28,28

#get the class
generative_adversarial_network=gans()

#Implement the gan
generative_adversarial_network.\
generative_adversarial_training_process(attributes,100, 784,\
original_image_format)
