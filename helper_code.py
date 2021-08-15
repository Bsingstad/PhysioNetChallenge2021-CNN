#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.

from operator import imod
import os, numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from scipy.signal import filtfilt, iirnotch, freqz, butter
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks
from scipy import signal

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# (Re)sort leads using the standard order of leads for the standard twelve-lead ECG.
def sort_leads(leads):
    x = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    leads = sorted(leads, key=lambda lead: (x.index(lead) if lead in x else len(x) + leads.index(lead)))
    return tuple(leads)

# Find header and recording files.
def find_challenge_files(data_directory):
    header_files = list()
    recording_files = list()
    for f in os.listdir(data_directory):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.hea':
            header_file = os.path.join(data_directory, root + '.hea')
            recording_file = os.path.join(data_directory, root + '.mat')
            if os.path.isfile(header_file) and os.path.isfile(recording_file):
                header_files.append(header_file)
                recording_files.append(recording_file)
    return header_files, recording_files

# Load header file as a string.
def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header

# Load recording file as an array.
def load_recording(recording_file, header=None, leads=None, key='val'):
    from scipy.io import loadmat
    recording = loadmat(recording_file)[key]
    if header and leads:
        recording = choose_leads(recording, header, leads)
    return recording

# Choose leads from the recording file.
def choose_leads(recording, header, leads):
    num_leads = len(leads)
    num_samples = np.shape(recording)[1]
    chosen_recording = np.zeros((num_leads, num_samples), recording.dtype)
    available_leads = get_leads(header)
    for i, lead in enumerate(leads):
        if lead in available_leads:
            j = available_leads.index(lead)
            chosen_recording[i, :] = recording[j, :]
    return chosen_recording

# Get recording ID.
def get_recording_id(header):
    recording_id = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                recording_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return recording_id

# Get leads from header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)

# Get age from header.
def get_age(header):
    age = None
    for l in header.split('\n'):
        if l.startswith('#Age'):
            try:
                age = float(l.split(': ')[1].strip())
            except:
                age = float('nan')
    return age

# Get sex from header.
def get_sex(header):
    sex = None
    for l in header.split('\n'):
        if l.startswith('#Sex'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

# Get number of leads from header.
def get_num_leads(header):
    num_leads = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_leads = float(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_leads

# Get frequency from header.
def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency

# Get number of samples from header.
def get_num_samples(header):
    num_samples = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_samples = float(l.split(' ')[3])
            except:
                pass
        else:
            break
    return num_samples

# Get analog-to-digital converter (ADC) gains from header.
def get_adc_gains(header, leads):
    adc_gains = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    adc_gains[j] = float(entries[2].split('/')[0])
                except:
                    pass
        else:
            break
    return adc_gains

# Get baselines from header.
def get_baselines(header, leads):
    baselines = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    baselines[j] = float(entries[4].split('/')[0])
                except:
                    pass
        else:
            break
    return baselines

# Get labels from header.
def get_labels(header):
    labels = list()
    scored_labels = np.asarray(pd.read_csv("dx_mapping_scored.csv").iloc[:,1], dtype="str")
    for l in header.split('\n'):
        if l.startswith('#Dx'):
            try:
                entries = l.split(': ')[1].split(',')
                for entry in entries:
                    if any(j == entry for j in scored_labels):
                        labels.append(entry.strip())
            except:
                pass
    return labels

# Save outputs from model.
def save_outputs(output_file, recording_id, classes, labels, probabilities):
    # Format the model outputs.
    recording_string = '#{}'.format(recording_id)
    class_string = ','.join(str(c) for c in classes)
    label_string = ','.join(str(l) for l in labels)
    probabilities_string = ','.join(str(p) for p in probabilities)
    output_string = recording_string + '\n' + class_string + '\n' + label_string + '\n' + probabilities_string + '\n'

    # Save the model outputs.
    with open(output_file, 'w') as f:
        f.write(output_string)

# Load outputs from model.
def load_outputs(output_file):
    with open(output_file, 'r') as f:
        for i, l in enumerate(f):
            if i==0:
                recording_id = l[1:] if len(l)>1 else None
            elif i==1:
                classes = tuple(entry.strip() for entry in l.split(','))
            elif i==2:
                labels = tuple(entry.strip() for entry in l.split(','))
            elif i==3:
                probabilities = tuple(float(entry) if is_finite_number(entry) else float('nan') for entry in l.split(','))
            else:
                break
    return recording_id, classes, labels, probabilities

#-----------------------------------------------------------#
#                                                           #
#                    My functions                           #
#                                                           #
#-----------------------------------------------------------#

def abbrev(snomed_classes):
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