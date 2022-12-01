import boto3
import librosa
import math
import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

----------------------------------------------
#how to access files on s3

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket = s3.Bucket("music-wav-charts")

i = 0
for obj in bucket.objects.all(): #resource
    if(obj.key[len(obj.key)-3:]) == "png":
        print(obj.key)
        i+=1
    #file = s3_client.download_file('music-wav-files', obj.key, 'audio.wav')
    #audio, sfreq = librosa.load('audio.wav')
    #print(audio)
print(i)
---------------------------------------------
#how to access files on s3

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket = s3.Bucket("music-wav-charts-2")

i = 0
for obj in bucket.objects.all(): #resource
    if(obj.key[len(obj.key)-3:]) == "wav":
        print(obj.key)
        i+=1
    #file = s3_client.download_file('music-wav-files', obj.key, 'audio.wav')
    #audio, sfreq = librosa.load('audio.wav')
    #print(audio)
print(i)
------------------------------------------------
#USED TO CREATE THE DATASET

SAMPLE_RATE = 22050
DURATION = 30 #measured in seconds

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket = s3.Bucket("music-wav-files")

SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
def save_mfcc(n_mfcc=13, n_fft=2048, 
              hop_length=512, num_segments = 5):
    
    with open('data.json', 'w') as fp:
        #dictionary to store data
        data = {
            "mapping": ["blues", "classical", "country", "disco",
                       "hiphop", "metal", "pop", "reggae", "rock"],
            "mfcc": [],
            "labels": []
        }

        num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        expected_num_mfcc_vectors_per_seg = math.ceil(num_samples_per_segment / hop_length)
        j = 1
        k = 1
        for obj in bucket.objects.all(): #resource
            if j == 100:
                k+=1
                j = 1
                
            if(obj.key[len(obj.key)-3:]) == "wav":
                file = s3_client.download_file('music-wav-files', obj.key, 'audio.wav')
                print("\nProcessing {}".format(obj.key))


                #load audio file
                signal, sr = librosa.load('audio.wav', sr=SAMPLE_RATE)

                #process segments extracting mfcc/storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                               sr = sr,
                                               n_fft = n_fft,
                                               n_mfcc = n_mfcc,
                                               hop_length=hop_length
                                               )

                    mfcc = mfcc.T

                    #store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_seg:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(k - 1)
                        print("{}, segment: {}".format(obj.key, s))

                j+=1


        json.dump(data, fp)

        return data


mfcc_dataset = save_mfcc(num_segments=10)
print("done processing")
----------------------------------------
#load the dataset
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
        
    X = np.array(data["mfcc"]) #X is the MFCC - input
    Y = np.array(data["labels"]) #Y is the labels (0-8) - output. 
    return X,Y #inputs, targets
---------------------------------------
def prepare_datasets(test_size, validation_size):
    #load data
    X, Y = load_data('data.json')
    
    #create the train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    
    #create the train/validation split
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_size)

    #3d array -> (130, 13, 1) see 12:00
    X_train = X_train[...,np.newaxis] #4d array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[...,np.newaxis]
    X_test = X_test[...,np.newaxis]
    
    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test
-------------------------------------------
def build_model(input_shape):
    #create model that you will pass the mfccs arrays into. once you pass them through these layers,
    #you should get a 1d array which allows you to map a given mfcc array to a music genre
    model = keras.Sequential()
    
    #1st conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', 
                        input_shape=input_shape)) 
    #32 filters, kernel grid size: 3x3, relu activation function, same input_shape as X_train
    
    model.add(keras.layers.MaxPool2D( (3,3), strides=(2,2), padding='same' ))
    #we maxpool so we can immediately downsample the input
    #stride is the rate at which the neural network moves over the MFCCs
    
    model.add(keras.layers.BatchNormalization())
    #a process that normalizes the activations in the current layer and the applications
    #that gets present to the subsequent layer. speeds up the model much faster because the numbers
    #are smaller.
    
    #2nd conv layer - same as the 1st
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', 
                                 input_shape=input_shape))
    
    model.add(keras.layers.MaxPool2D( (3,3), strides=(2,2), padding='same' ))
    model.add(keras.layers.BatchNormalization())
    
    
    #3rd conv layer
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', 
                                 input_shape=input_shape))
    #we want to change the kernel to 2x2
    
    model.add(keras.layers.MaxPool2D( (2,2), strides=(2,2), padding='same' ))
    #we want to change the stride to 2x2
    
    model.add(keras.layers.BatchNormalization())
    
    #flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    
    #output layer - want to flatten the 2d array into a 1d array.
    #the 1d array is the mapping that associates a given mfcc array to a specific genre, as discussed
    #in the dataset design section in this project's report
    model.add(keras.layers.Dense(10,activation='softmax'))
    #we will be returning this 1d array
    
    return model
----------------------------------------------
correct = 0

def predict(model, X, Y):
    global correct
    X = X[np.newaxis, ...]
    
    #prediction = [ [0.1, 0.2, ...] ]
    prediction = model.predict(X) #X -> (1, 130, 13, 1)
    
    #extract index (genre)
    predicted_index = np.argmax(prediction, axis=1) # [3]
    print("Expected index: {}, Predicted index: {}".format(
    Y, predicted_index))
    
    if predicted_index == Y:
        correct+=1
    
    
if __name__ == "__main__":
    #create train, validation, and test sets
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_datasets(0.25, 0.2)
    #X_train trains the dataset. 25% of data.json is used for this
    #X_validate verifies that the information is accurate. 20% of the data is used for this.
    #X_test is to verify the model works. 55% of the data is used for this.

    #build the CNN network
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) 
    #X_train.shape[0] is the number of samples.
    #.shape[1] is the number of time bins
    #.shape[2] is the number of MFCCs
    #.shape[3] is the number of channels/depth
    
    model = build_model(input_shape)
    
    #compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    #train the CNN
    model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
              batch_size=32,
              epochs=30 #train the model for 30 iterations
             )
    
    #evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, Y_test,
                                              verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    
    
    #make predictions on a sample
    for i in range(100, 2248):
        X = X_test[i]
        Y = Y_test[i]
        predict(model, X,Y)
    print("Accuracy Rate: " + str(correct/2147) )

    #The accuracy rate is 67%. For a model which uses less than 50% of a dataset to train and validate,
    #this is pretty good.

---------------------------------------------------
