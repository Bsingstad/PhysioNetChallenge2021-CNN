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
from scipy.io import loadmat
import pandas as pd
# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

other_diag = ["bundle branch block","bradycardia","1st degree av block", "incomplete right bundle branch block", "left axis deviation", "left anterior fascicular block", "left bundle branch block", "low qrs voltages",
        "nonspecific intraventricular conduction disorder", "poor R wave Progression", "prolonged pr interval", "prolonged qt interval", "qwave abnormal", "right axis deviation", "right bundle branch block", "t wave abnormal", "t wave inversion"]


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
    num_classes_all = len(classes)

    with open('classes.txt', 'w') as f:
        for class_ in classes:
            f.write("%s\n" % class_)
    f.close()

    print('Number of classes (all) = ', num_classes_all)

    SNOMED_scored=pd.read_csv("./dx_mapping_scored.csv", sep=",")
    lab_arr = np.asarray(SNOMED_scored['SNOMEDCTCode'], dtype="str")
    scored_classes = []
    for i in classes:
        for j in lab_arr:
            if i == '':
                continue
            if i == j:
                scored_classes.append(i)
    scored_classes = sorted(scored_classes)
    num_classes = len(scored_classes)

    print('Number of scored classes = ', num_classes)

    labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes

    for i in range(len(recording_files)):
        current_labels = get_labels(load_header(recording_files[i].replace('.mat','.hea')))
        for label in current_labels:
            if label in scored_classes:
                j = scored_classes.index(label)
                labels[i, j] = 1
    labels = labels *1

    abbr = abbrev(scored_classes)
    # Extract the features and labels from the dataset.
    print('Calculate heart rate from lead-II...')

    heart_rates = calc_hr(recording_files)

    # Train a model for each lead set.
    for leads in lead_sets:
        print('Training model for {}-lead set: {}...'.format(len(leads), ', '.join(leads)))
        
        regelmessig_ind, uregelmessig_ind = regelmessigVSuregelmessig(labels, abbr)

        ureg_vs_reg_data = np.concatenate([np.asarray(recording_files)[regelmessig_ind],np.asarray(recording_files)[uregelmessig_ind]])
        ureg_vs_reg_label = np.concatenate([np.ones(len(regelmessig_ind)),np.zeros(len(uregelmessig_ind))])

        #shuffle data:
        index_array = np.arange(len(ureg_vs_reg_label))
        np.random.shuffle(index_array)
        ureg_vs_reg_data = ureg_vs_reg_data[index_array]
        ureg_vs_reg_label = ureg_vs_reg_label[index_array]

        ureg_vs_reg_label = np.expand_dims(ureg_vs_reg_label, axis=1)


        hr_data = np.concatenate([heart_rates[regelmessig_ind],heart_rates[uregelmessig_ind]])
        hr_data = np.expand_dims(hr_data,axis=1)

        # Train rythm model
        batch_size = 30 # change this to 30!!!
        num_leads = len(leads)
        epochs = 3
        signal_len = 5000
        train_rythm_model_2(ureg_vs_reg_data, hr_data, ureg_vs_reg_label, num_leads,batch_size, epochs, signal_len, model_name = model_directory + "/" + str(num_leads) + "_leads_rythm")

        # Train regular rythm model
        regl_rytm_data , ohe_reg_rythm_classes =regelmessige_diag(np.asarray(recording_files) ,labels, abbr)
        #shuffle data:
        index_array_2 = np.arange(len(ohe_reg_rythm_classes))
        np.random.shuffle(index_array_2)
        ohe_reg_rythm_classes = ohe_reg_rythm_classes[index_array_2]
        regl_rytm_data = regl_rytm_data[index_array_2]

        batch_size = 30
        epochs = 5
        signal_len = 5000
        train_rythm_model_2(regl_rytm_data, hr_data ,ohe_reg_rythm_classes, num_leads,batch_size, epochs, signal_len, model_name = model_directory + "/" + str(num_leads) + "_leads_regular_rythm")

        # Train irregular rythm model
        uregl_rytm_data , ohe_ureg_rythm_classes =uregelmessige_diag(np.asarray(recording_files) ,labels, abbr)
        #shuffle data:
        index_array_3 = np.arange(len(ohe_ureg_rythm_classes))
        np.random.shuffle(index_array_3)
        ohe_ureg_rythm_classes = ohe_ureg_rythm_classes[index_array_3]
        uregl_rytm_data = uregl_rytm_data[index_array_3]

        batch_size = 30
        epochs = 5
        signal_len = 5000
        train_rythm_model_2(uregl_rytm_data,hr_data,ohe_ureg_rythm_classes, num_leads,batch_size, epochs, signal_len, model_name = model_directory + "/" +str(num_leads) + "_leads_irregular_rhythm")

        labels_rytme = rytme_labels(labels=labels ,abbr=abbr)

        cc_data = labels_rytme
        for i in other_diag:
            new_diag = finn_diagnoser(labels,abbr,i)
            batch_size = 30
            epochs = 5
            signal_len = 5000
            name = str(num_leads) + "_leads_" + i
            train_classifier_chain(np.asarray(recording_files),cc_data,new_diag, num_leads,batch_size, epochs, signal_len,model_name = model_directory + "/" + name)
            cc_data = np.concatenate((cc_data,new_diag),axis=1)


################################################################################
#
# Running trained model function
#
################################################################################

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def run_model(model, header, recording):
    num_leads = int(list(model.keys())[0].split("_")[0])

    temp_classes = []
    with open('classes.txt', 'r') as f:
        for line in f:
            temp_classes.append(line)

    classes=[s.strip('\n') for s in temp_classes]

    probabilities = np.zeros(len(classes))

    model_list = ["rythm","regular_rythm", "irregular_rhythm", "bundle branch block","bradycardia","1st degree av block", 
    "incomplete right bundle branch block", "left axis deviation", "left anterior fascicular block", 
    "left bundle branch block", "low qrs voltages","nonspecific intraventricular conduction disorder", 
    "poor R wave Progression", "prolonged pr interval", "prolonged qt interval", "qwave abnormal", 
    "right axis deviation", "right bundle branch block", "t wave abnormal", "t wave inversion"]
    
    heart_rate_val = calc_hr_predict(recording,header)
    heart_rate_val = np.expand_dims(heart_rate_val,axis=0)

    fft_data = fourier_trans_ecg(recording,header,num_leads)

    rythm = model[str(num_leads) + "_leads_" + model_list[0] + ".h5"].predict([np.expand_dims(fft_data,axis=0),np.expand_dims(heart_rate_val,axis=0)])
    if rythm > 0.5:
        probabilities[:6] = model[str(num_leads) + "_leads_" + model_list[1] + ".h5"].predict([np.expand_dims(fft_data,axis=0),np.expand_dims(heart_rate_val,axis=0)])   
    elif rythm <0.5:
        probabilities[6:9] = model[str(num_leads) + "_leads_" + model_list[2] + ".h5"].predict([np.expand_dims(fft_data,axis=0),np.expand_dims(heart_rate_val,axis=0)])
    
    raw_ecg = preprocess_ecg(recording,header,num_leads)
    for i,j in enumerate(model_list[3:]):
        model_name = str(num_leads) + "_leads_" + j + ".h5"
        probabilities[9+i] = model[model_name].predict([np.expand_dims(raw_ecg,axis=0),np.expand_dims(probabilities[:9+i],axis=0)])




    threshold = np.ones(len(classes))*0.5 # should make a better threshold here
    binary_prediction = probabilities > threshold
    binary_prediction = binary_prediction * 1
    binary_prediction = np.asarray(binary_prediction, dtype=np.int).ravel()
    probabilities = np.asarray(probabilities, dtype=np.float32).ravel()
    return classes, binary_prediction, probabilities

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
    model_list = ["rythm","regular_rythm", "irregular_rhythm", "bundle branch block","bradycardia","1st degree av block", 
    "incomplete right bundle branch block", "left axis deviation", "left anterior fascicular block", 
    "left bundle branch block", "low qrs voltages","nonspecific intraventricular conduction disorder", 
    "poor R wave Progression", "prolonged pr interval", "prolonged qt interval", "qwave abnormal", 
    "right axis deviation", "right bundle branch block", "t wave abnormal", "t wave inversion"]
    classifier_chain = {}
    for i in model_list:
        model_name = str(len(leads)) + "_leads_" + i + ".h5"
        model = tf.keras.models.load_model(os.path.join(model_directory, model_name))
        classifier_chain[model_name] = model
    return classifier_chain

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

