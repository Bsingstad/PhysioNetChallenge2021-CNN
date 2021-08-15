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

    abbr = abbreviation(scored_classes)
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
        epochs = 1
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
        epochs = 1
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
        epochs = 1
        signal_len = 5000
        train_rythm_model_2(uregl_rytm_data,hr_data,ohe_ureg_rythm_classes, num_leads,batch_size, epochs, signal_len, model_name = model_directory + "/" +str(num_leads) + "_leads_irregular_rhythm")

        labels_rytme = rytme_labels(labels=labels ,abbr=abbr)

        cc_data = labels_rytme
        for i in other_diag:
            new_diag = finn_diagnoser(labels,abbr,i)
            batch_size = 30
            epochs = 1
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

#-----------------------------------------------------------#
#                                                           #
#                    My functions                           #
#                                                           #
#-----------------------------------------------------------#

def abbreviation(snomed_classes):
    SNOMED_scored = pd.read_csv("./dx_mapping_scored.csv", sep=",")
    snomed_abbr = []
    for j in range(len(snomed_classes)):
        for i in range(len(SNOMED_scored.iloc[:,1])):
            if (str(SNOMED_scored.iloc[:,1][i]) == snomed_classes[j]):
                snomed_abbr.append(SNOMED_scored.iloc[:,0][i])
                
    snomed_abbr = np.asarray(snomed_abbr)
    return snomed_abbr

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

def pan_tompkins(data, fs):
    lowcut = 5.0
    highcut = 15.0
    filter_order = 2
    nyquist_freq = 0.5 * fs

    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq

    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)

    diff_y = np.ediff1d(y)
    squared_diff_y=diff_y**2
    integrated_squared_diff_y =np.convolve(squared_diff_y,np.ones(5))

    max_h = integrated_squared_diff_y.max()

    peaks=find_peaks(integrated_squared_diff_y,height=max_h/2, distance=fs/3)

    hr = np.nanmean(60 /(np.diff(peaks[0])/fs)).mean()


    return hr

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def calc_hr(ecg_filenames):
    heart_rate = np.zeros(len(ecg_filenames))
    for i,j in enumerate(ecg_filenames):
        data, head = load_challenge_data(j)
        heart_rate[i] = pan_tompkins(data[1],int(head[0].split(" ")[2]))
    heart_rate[np.where(np.isnan(heart_rate))[0]] = np.nanmean(heart_rate)
    return heart_rate

def calc_hr_predict(data,header):
    heart_rate = pan_tompkins(data[1],int(header.split(" ")[2]))
    if heart_rate == "NaN":
        heart_rate = 80
    return heart_rate


def regelmessigVSuregelmessig(labels, abbr):
    atrieflutter = np.where(labels[:,np.where(abbr == 'atrial flutter')[0]] == 1)[0]
    pacerythm = np.where(labels[:,np.where(abbr == 'pacing rhythm')[0]] == 1)[0]
    sinus = np.where(labels[:,np.where(abbr == 'sinus rhythm')[0]] == 1)[0]
    sinus_brad = np.where(labels[:,np.where(abbr == 'sinus bradycardia')[0]] == 1)[0]
    sinus_tach = np.where(labels[:,np.where(abbr == 'sinus tachycardia')[0]] == 1)[0]
    sinus_arr = np.where(labels[:,np.where(abbr == 'sinus arrhythmia')[0]] == 1)[0]
    #------------------------------------------------------------------------------
    regelmessig = np.unique(np.concatenate([atrieflutter,pacerythm,sinus,sinus_brad,sinus_tach,sinus_arr]))
    #------------------------------------------------------------------------------
    afib = np.where(labels[:,np.where(abbr == 'atrial fibrillation')[0]] == 1)[0]
    VES1 = np.where(labels[:,np.where(abbr == 'ventricular premature beats')[0]] == 1)[0]
    VES2 = np.where(labels[:,np.where(abbr == 'premature ventricular contractions')[0]] == 1)[0]
    VES = np.concatenate([VES1,VES2])
    SVES1 = np.where(labels[:,np.where(abbr == 'supraventricular premature beats')[0]] == 1)[0]
    SVES2 = np.where(labels[:,np.where(abbr == 'premature atrial contraction')[0]] == 1)[0]
    SVES = np.concatenate([SVES1,SVES2])
    #------------------------------------------------------------------------------
    uregelmessig = np.unique(np.concatenate([afib,VES,SVES]))
    #------------------------------------------------------------------------------
    del_ureg, del_reg = np.intersect1d(uregelmessig,regelmessig,return_indices=True)[1:]

    uregelmessig = np.delete(uregelmessig, del_ureg)
    regelmessig = np.delete(regelmessig, del_reg)
    return regelmessig, uregelmessig


def rytme_labels(labels, abbr):
    rythm_labels = labels[:,[int(np.where(abbr == 'atrial flutter')[0]),int(np.where(abbr == 'pacing rhythm')[0]), int(np.where(abbr == 'sinus rhythm')[0]),
            int(np.where(abbr == 'sinus bradycardia')[0]), int(np.where(abbr == 'sinus tachycardia')[0]),int(np.where(abbr == 'sinus arrhythmia')[0]),
            int(np.where(abbr == 'atrial fibrillation')[0]), int(np.where(abbr == 'ventricular premature beats')[0]) | int(np.where(abbr == 'premature ventricular contractions')[0]),
            int(np.where(abbr == 'supraventricular premature beats')[0]) | int(np.where(abbr == 'premature atrial contraction')[0])]]
    return rythm_labels

def finn_diagnoser(labels, abbr, navn):
    arr = np.zeros((labels.shape[0],1))
    idx = np.where(labels[:,np.where(abbr == navn)[0]] == 1)[0]
    arr[idx] = 1
    return arr



def regelmessige_diag(ecg_filenames,labels, abbr):
  atrieflutter = np.where(labels[:,np.where(abbr == 'atrial flutter')[0]] == 1)[0]
  pacerythm = np.where(labels[:,np.where(abbr == 'pacing rhythm')[0]] == 1)[0]
  sinus = np.where(labels[:,np.where(abbr == 'sinus rhythm')[0]] == 1)[0]
  sinus_brad = np.where(labels[:,np.where(abbr == 'sinus bradycardia')[0]] == 1)[0]
  sinus_tach = np.where(labels[:,np.where(abbr == 'sinus tachycardia')[0]] == 1)[0]
  sinus_arr = np.where(labels[:,np.where(abbr == 'sinus arrhythmia')[0]] == 1)[0]

  regelmessige_rytmer = np.concatenate([ecg_filenames[atrieflutter],ecg_filenames[pacerythm],
                                    ecg_filenames[sinus],ecg_filenames[sinus_brad],
                                    ecg_filenames[sinus_tach],ecg_filenames[sinus_arr]
                                    ])
  
  regelmessige_rythm_classes = np.zeros((len(regelmessige_rytmer),len([ecg_filenames[atrieflutter],ecg_filenames[pacerythm],
                                    ecg_filenames[sinus],ecg_filenames[sinus_brad],
                                    ecg_filenames[sinus_tach],ecg_filenames[sinus_arr]])))
  counter = 0
  for i,j in enumerate([atrieflutter,pacerythm,sinus,sinus_brad, sinus_tach, sinus_arr]):
    for k in j:

      regelmessige_rythm_classes[counter,i] = 1
      counter += 1

  return regelmessige_rytmer, regelmessige_rythm_classes

def uregelmessige_diag(ecg_filenames, labels, abbr):
  afib = np.where(labels[:,np.where(abbr == 'atrial fibrillation')[0]] == 1)[0]
  VES1 = np.where(labels[:,np.where(abbr == 'ventricular premature beats')[0]] == 1)[0]
  VES2 = np.where(labels[:,np.where(abbr == 'premature ventricular contractions')[0]] == 1)[0]
  VES = np.concatenate([VES1,VES2])
  SVES1 = np.where(labels[:,np.where(abbr == 'supraventricular premature beats')[0]] == 1)[0]
  SVES2 = np.where(labels[:,np.where(abbr == 'premature atrial contraction')[0]] == 1)[0]
  SVES = np.concatenate([SVES1,SVES2])

  uregelmessige_rytmer = np.concatenate([ecg_filenames[afib],ecg_filenames[VES],
                                    ecg_filenames[SVES]])
  
  uregelmessige_rythm_classes = np.zeros((len(uregelmessige_rytmer),len([ecg_filenames[afib],ecg_filenames[VES],
                                    ecg_filenames[SVES]])))
  
  counter = 0
  for i,j in enumerate([afib,VES,SVES]):
    for k in j:

      uregelmessige_rythm_classes[counter,i] = 1
      counter += 1

  return uregelmessige_rytmer, uregelmessige_rythm_classes


def train_rythm_model_2(data, cc_data, labels,num_leads,batch_size, epochs, signal_len, model_name):
  model = encoder_bin_2((signal_len,num_leads),cc_data.shape[1],labels.shape[1])
  model.fit(x=batch_generator_2(batch_size=batch_size,  gen_x=generate_X_fourier(data, num_leads), gen_x2=generate_cc_data(cc_data), 
                              gen_y=generate_y(labels), num_leads=num_leads,num_classes=labels.shape[1],cc_data_len=cc_data.shape[1]),
                              epochs=epochs, 
                              steps_per_epoch=(len(data)/batch_size))
  model.save(model_name + ".h5")

def train_classifier_chain(data, cc_data, labels,num_leads,batch_size, epochs, signal_len, model_name):
  model = encoder_bin_2((signal_len,num_leads),cc_data.shape[1],labels.shape[1])
  model.fit(x=batch_generator_2(batch_size=batch_size,  gen_x=generate_X_rawecg(data, num_leads), gen_x2=generate_cc_data(cc_data), 
                              gen_y=generate_y(labels), num_leads=num_leads,num_classes=labels.shape[1],cc_data_len=cc_data.shape[1]),
                              epochs=epochs, 
                              steps_per_epoch=(len(data)/batch_size))
  model.save(model_name + ".h5")


def fourier_trans_ecg(data, header, num_leads):

  if num_leads == 12:
    ecg = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
  elif num_leads == 6:
    ecg = data[[0,1,2,3,4,5]]
  elif num_leads == 4:
    ecg = data[[0,1,2,7]]
  elif num_leads == 3:
    ecg = data[[0,1,7]]
  elif num_leads == 2:
    ecg = data[[0,1]]

  samp_freq = int(header.split(" ")[2])
  cutoff = 0.1 # remove noise at 0Hz
  T = 1.0 / samp_freq
  N = ecg.shape[1]
  fourier_sig = np.ones([num_leads,5000])
  for i,j in enumerate(ecg):
      #filt_ecg = butter_highpass_filter(j,cutoff,samp_freq)
      filt_ecg = j
      yf = fft(filt_ecg)
      xf = fftfreq(N,T)[:N//2]
      
      fft_res = 2.0/N * np.abs(yf[0:N//2])
      freq_100 = fft_res[:np.where(xf >= 100)[0][0]]
      #down sample/upsample fourier transformed signal to length 5000
      dwn_smp_sig = signal.resample(freq_100,int(N/(N/5000)))
      fourier_sig[i,:dwn_smp_sig.shape[0]] = dwn_smp_sig

  fourier_sig = fourier_sig.reshape(fourier_sig.shape[1],fourier_sig.shape[0])
  return fourier_sig

def preprocess_ecg(data, header, num_leads):
  if num_leads == 12:
      data = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
  elif num_leads == 6:
      data = data[[0,1,2,3,4,5]]
  elif num_leads == 4:
      data = data[[0,1,2,7]]
  elif num_leads == 3:
      data = data[[0,1,7]]
  elif num_leads == 2:
      data = data[[0,1]]

  if int(header.split(" ")[2]) != 500:
      data_new = np.ones([num_leads,int((int(header.split(" ")[3])/int(header.split(" ")[2]))*500)])
      for i,j in enumerate(data):
          data_new[i] = signal.resample(j, int((int(header.split(" ")[3])/int(header.split(" ")[2]))*500))
      data = data_new
  data = pad_sequences(data, maxlen=5000, truncating='post',padding="post")
  #data = data + np.random.choice([0,0,0,np.random.rand(12,5000)*random.randint(0, 50)])
      
  data = data.reshape(data.shape[1],data.shape[0])
  return data

def batch_generator(batch_size, gen_x, gen_y, num_leads, num_classes): 
    #np.random.shuffle(order_array)
    batch_features = np.zeros((batch_size,5000, num_leads))
    batch_labels = np.zeros((batch_size,num_classes))
    while True:
        for i in range(batch_size):

            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
        yield batch_features, batch_labels    

def generate_y(y_train):
    while True:
        for i in y_train:
            yield i

def generate_X_rawecg(X_train_file, num_leads):
    while True:
        for h in X_train_file:
          data, header_data = load_challenge_data(h)
          if num_leads == 12:
              data = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
          elif num_leads == 6:
              data = data[[0,1,2,3,4,5]]
          elif num_leads == 4:
              data = data[[0,1,2,7]]
          elif num_leads == 3:
              data = data[[0,1,7]]
          elif num_leads == 2:
              data = data[[0,1]]

          if int(header_data[0].split(" ")[2]) != 500:
              data_new = np.ones([num_leads,int((int(header_data[0].split(" ")[3])/int(header_data[0].split(" ")[2]))*500)])
              for i,j in enumerate(data):
                  data_new[i] = signal.resample(j, int((int(header_data[0].split(" ")[3])/int(header_data[0].split(" ")[2]))*500))
              data = data_new
          data = pad_sequences(data, maxlen=5000, truncating='post',padding="post")
          #data = data + np.random.choice([0,0,0,np.random.rand(12,5000)*random.randint(0, 50)])
              
          data = data.reshape(data.shape[1],data.shape[0])
          yield data

def generate_X_fourier(X_train_file, num_leads):
    while True:
        for h in X_train_file:
          data, header_data = load_challenge_data(h)
          
          if num_leads == 12:
            ecg = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
          elif num_leads == 6:
            ecg = data[[0,1,2,3,4,5]]
          elif num_leads == 4:
            ecg = data[[0,1,2,7]]
          elif num_leads == 3:
            ecg = data[[0,1,7]]
          elif num_leads == 2:
            ecg = data[[0,1]]

          samp_freq = int(header_data[0].split(" ")[2])
          cutoff = 0.1 # remove noise at 0Hz
          T = 1.0 / samp_freq
          N = ecg.shape[1]
          fourier_sig = np.ones([num_leads,5000])
          for i,j in enumerate(ecg):
              #filt_ecg = butter_highpass_filter(j,cutoff,samp_freq)
              filt_ecg = j
              yf = fft(filt_ecg)
              xf = fftfreq(N,T)[:N//2]
              
              fft_res = 2.0/N * np.abs(yf[0:N//2])
              freq_100 = fft_res[:np.where(xf >= 100)[0][0]]
              #down sample/upsample fourier transformed signal to length 5000
              dwn_smp_sig = signal.resample(freq_100,int(N/(N/5000)))
              fourier_sig[i,:dwn_smp_sig.shape[0]] = dwn_smp_sig

          fourier_sig = fourier_sig.reshape(fourier_sig.shape[1],fourier_sig.shape[0])
          yield fourier_sig

def batch_generator_2(batch_size, gen_x, gen_x2, gen_y, num_leads, num_classes, cc_data_len): 
    #np.random.shuffle(order_array)
    batch_cc = np.zeros((batch_size, cc_data_len))
    batch_features = np.zeros((batch_size,5000, num_leads))
    batch_labels = np.zeros((batch_size,num_classes))
    while True:
        for i in range(batch_size):
            batch_cc[i] = next(gen_x2)
            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
        batch_features_comb = [batch_features, batch_cc]
        yield batch_features_comb, batch_labels  

def generate_cc_data(data):
    while True:
        for i in data:
            yield i


def encoder_bin_2(n_input_1, n_input_2, n_output):
    input_1=tf.keras.layers.Input(shape=(n_input_1))
    input_2=tf.keras.layers.Input(shape=(n_input_2))
     # conv block -1
    conv1 = tf.keras.layers.Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(input_1)
    conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv1 = tf.keras.layers.PReLU(shared_axes=[1])(conv1)
    conv1 = tf.keras.layers.SpatialDropout1D(rate=0.2)(conv1)
    conv1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
    # conv block -2
    conv2 = tf.keras.layers.Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(conv1)
    conv2 = tfa.layers.InstanceNormalization()(conv2)
    conv2 = tf.keras.layers.PReLU(shared_axes=[1])(conv2)
    conv2 = tf.keras.layers.SpatialDropout1D(rate=0.2)(conv2)
    conv2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
    # conv block -3
    conv3 = tf.keras.layers.Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(conv2)
    conv3 = tfa.layers.InstanceNormalization()(conv3)
    conv3 = tf.keras.layers.PReLU(shared_axes=[1])(conv3)
    conv3 = tf.keras.layers.Dropout(rate=0.2)(conv3)
    # split for attention
    attention_data = tf.keras.layers.Lambda(lambda x: x[:,:,:])(conv3)
    attention_softmax = tf.keras.layers.Lambda(lambda x: x[:,:,:])(conv3)
    # attention mechanism
    attention_softmax = tf.keras.layers.Softmax()(attention_softmax)
    multiply_layer = tf.keras.layers.Multiply()([attention_softmax,attention_data])
    # last layer
    dense_layer = tf.keras.layers.Dense(units=512,activation='relu')(multiply_layer)
    dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
    # output layer
    flatten_layer = tf.keras.layers.Flatten()(dense_layer)

    mod1 = tf.keras.Model(inputs=input_1, outputs=flatten_layer)

    mod2 = tf.keras.layers.Dense(n_input_2, activation="relu")(input_2)
    mod2 = keras.models.Model(inputs=input_2, outputs=mod2)

    combined = keras.layers.concatenate([mod1.output, mod2.output]) 

    output_layer = tf.keras.layers.Dense(units=n_output,activation='sigmoid')(combined)

    model = tf.keras.models.Model(inputs=[mod1.input, mod2.input], outputs=output_layer)

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model