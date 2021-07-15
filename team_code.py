#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of the required functions, remove non-required functions, and add your own functions.

################################################################################
#
# Imported functions and variables
#
################################################################################

# Import functions. These functions are not required. You can change or remove them.
from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
import pandas as pd
# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

################################################################################
#
# Training model function
#
################################################################################

# Train your model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract the classes from the dataset.
    print('Extracting classes...')

    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically if not numbers.
    num_classes = len(classes)

    with open('classes.txt', 'w') as f:
        for class_ in classes:
            f.write("%s\n" % class_)
    f.close()

    print('Number of classes = ', num_classes)

    # Extract the features and labels from the dataset.
    print('Extracting features and labels...')

    #data = np.zeros((num_recordings, 14), dtype=np.float32) # 14 features: one feature for each lead, one feature for age, and one feature for sex
    #labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes
    '''
    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])

        # Get age, sex and root mean square of the leads.
        age, sex, rms = get_features(header, recording, twelve_leads)
        data[i, 0:12] = rms
        data[i, 12] = age
        data[i, 13] = sex

        current_labels = get_labels(header)
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                labels[i, j] = 1
    '''
    def generate_y(header_files, classes):
        while True:
            for i in order_array:
                labels = np.zeros((len(classes)), dtype=np.bool)
                head = load_header(header_files[i])
                current_labels = get_labels(head)
                for label in current_labels:
                    if label in classes:
                        j = classes.index(label)
                        labels[j] = 1
                yield labels

    def generate_X(recording_files, num_leads):
        while True:
            for i in order_array:
                recording = load_recording(recording_files[i])
                X_train = keras.preprocessing.sequence.pad_sequences(recording, maxlen=5000, truncating='post',padding="post")
                if num_leads == 12:
                    X_train = X_train
                elif num_leads == 6:
                    X_train = X_train[[0,1,2,3,4,5]]
                elif num_leads == 4:
                    X_train = X_train[[0,1,2,7]]
                elif num_leads == 3:
                    X_train = X_train[[0,1,7]]
                elif num_leads == 2:
                    X_train = X_train[[0,1]]                      
                X_train = X_train.reshape(5000,num_leads)
                yield X_train
    
    def batch_generator(batch_size, gen_x, gen_y, classes, num_leads):
        np.random.shuffle(order_array)
        batch_features = np.zeros((batch_size,5000, num_leads))
        batch_labels = np.zeros((batch_size,len(classes)))
        while True:
            for i in range(batch_size):

                batch_features[i] = next(gen_x)
                batch_labels[i] = next(gen_y)

            yield batch_features, batch_labels


    # Train a model for each lead set.
    for leads in lead_sets:
        print('Training model for {}-lead set: {}...'.format(len(leads), ', '.join(leads)))

        batchsize = 30
        model = create_model(len(leads), len(classes))
        global order_array
        order_array = np.arange(0,num_recordings,1)


        model.fit(x=batch_generator(batch_size=batchsize, gen_x=generate_X(recording_files, len(leads)), gen_y=generate_y(header_files, classes), classes=classes, num_leads=len(leads)), 
            epochs=4, steps_per_epoch=(num_recordings/batchsize) 
            #,class_weight=class_dict, callbacks=[lr_schedule]
            )
        filename = os.path.join(model_directory, 'model_{}_leads.h5'.format(len(leads)))
        model.save(filename)

        # Define parameters for random forest classifier.
        #n_estimators = 3     # Number of trees in the forest.
        #max_leaf_nodes = 100 # Maximum number of leaf nodes in each tree.
        #random_state = 123   # Random state; set for reproducibility.

        # Extract the features for the model.
        #feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
        #features = data[:, feature_indices]

        # Train the model.
        #imputer = SimpleImputer().fit(features)
        #features = imputer.transform(features)
        #classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

        # Save the model.
        #save_model(model_directory, leads, classes, imputer, classifier)

################################################################################
#
# Running trained model function
#
################################################################################

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def run_model(model, header, recording):
    #classes = model['classes']
    #leads = model.input.shape[2]
    #classifier = model['classifier']

    # Load features.
    #num_leads = len(leads)
    #data = np.zeros(num_leads+2)
    #age, sex, rms = get_features(header, recording, leads)
    #data[0:num_leads] = rms
    #data[num_leads] = age
    #data[num_leads+1] = sex

    # Impute missing data.
    #features = data.reshape(1, -1)
    #features = imputer.transform(features)

    # Predict labels and probabilities.
    temp_classes = []
    with open('classes.txt', 'r') as f:
        for line in f:
            temp_classes.append(line)

    classes=[s.strip('\n') for s in temp_classes]

    recording = keras.preprocessing.sequence.pad_sequences(recording, maxlen=5000, truncating='post',padding="post")
    probabilities = model.predict(np.expand_dims(recording.reshape(recording.shape[1],recording.shape[0]),0))
    threshold = np.ones(len(classes))*0.5 # should make a better threshold here
    labels = probabilities > threshold
    labels = labels * 1
    labels = np.asarray(labels, dtype=np.int).ravel()
    probabilities = np.asarray(probabilities, dtype=np.float32).ravel()
    return classes, labels, probabilities

################################################################################
#
# File I/O functions
#
################################################################################

# Save a trained model. This function is not required. You can change or remove it.
#def save_model(model_directory, leads, classes, imputer, classifier):
#    d = {'leads': leads, 'classes': classes, 'imputer': imputer, 'classifier': classifier}
#    filename = os.path.join(model_directory, get_model_filename(leads))
#    joblib.dump(d, filename, protocol=0)

# Load a trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def load_model(model_directory, leads):
    filename = os.path.join(model_directory, get_model_filename(leads))
    model = keras.models.load_model(filename)
    return model

# Define the filename(s) for the trained models. This function is not required. You can change or remove it.
def get_model_filename(leads):
    sorted_leads = sort_leads(leads)
    return 'model_{}_leads.h5'.format(len(sorted_leads))

################################################################################
#
# Feature extraction function
#
################################################################################

# Extract features from the header and recording. This function is not required. You can change or remove it.

def create_model(num_leads, num_classes):
    inputlayer = keras.layers.Input(shape=(5000, num_leads)) 

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=15,input_shape=(5000, num_leads), padding='same')(inputlayer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)
    #Legger til spatial dropout for å få med mer enn bare V4 som prediksjonsgrunnlag
    conv1 = keras.layers.SpatialDropout1D(0.1)(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=10, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    #Legger til spatial dropout for å få med mer enn bare V4 som prediksjonsgrunnlag
    conv2 = keras.layers.SpatialDropout1D(0.1)(conv2)

    conv3 = keras.layers.Conv1D(512, kernel_size=5,padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    #Legger til spatial dropout for å få med mer enn bare V4 som prediksjonsgrunnlag
    conv3 = keras.layers.Dropout(0.2)(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    #gap_layer = keras.layers.Flatten()(conv3)


    output_layer = tf.keras.layers.Dense(units=num_classes,activation='sigmoid', name='output_layer')(gap_layer)

    model = keras.Model(inputs=inputlayer, outputs=output_layer)
    

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), 
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', dtype=None, threshold=0.5)])
    return model
