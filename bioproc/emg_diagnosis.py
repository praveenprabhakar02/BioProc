"""
emg_diagnosis.py

Author: Praveen Prabhakar KR
Email: praveenp@msu.edu

Module contains the ANN functions for diagnosing EMG signals.
(For future development of the package).
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as k
import wfdb
from . import functions as fn
from . import emg as em

def emg_data():
    """
    Data for training the Neural Network.
    Data obtained from PhysioNet. 
    https://www.physionet.org/content/emgdb/1.0.0/
    """

    #healthy data
    emg_healthy = wfdb.rdsamp('./Examples/Training_data/emg_healthy')
    emg_healthy = emg_healthy[0]
    emg_healthy = emg_healthy[:,0]

    #myopathy data
    emg_myopathy = wfdb.rdsamp('./Examples/Training_data/emg_myopathy')
    emg_myopathy = emg_myopathy[0]
    emg_myopathy = emg_myopathy[:,0]
    
    #neuropathy data
    emg_neuropathy = wfdb.rdsamp('./Examples/Training_data/emg_neuropathy')
    emg_neuropathy = emg_neuropathy[0]
    emg_neuropathy = emg_neuropathy[:,0]

    return emg_healthy, emg_myopathy, emg_neuropathy


def training(emg_healthy, emg_myopathy, emg_neuropathy):
    """
    
    """

    healthy = emg_healthy.size
    myopathy = emg_myopathy.size
    neuropathy = emg_neuropathy.size

    healthy_train = int(0.8*healthy)
    myopathy_train = int(0.8*myopathy)
    neuropathy_train = int(0.8*neuropathy)
    healthy_test = healthy - healthy_train
    myopathy_test = myopathy - myopathy_train
    neuropathy_test = neuropathy - neuropathy_train

    train_data = np.concatenate((emg_healthy[0:healthy_train], emg_myopathy[0:myopathy_train], emg_neuropathy[0:neuropathy_train]))
    train_labels = np.chararray(healthy_train+myopathy_train+neuropathy_train, itemsize=10)
    train_labels[0:healthy_train] = 'healthy'
    train_labels[healthy_train:healthy_train+myopathy_train] = 'myopathy'
    train_labels[healthy_train+myopathy_train:healthy_train+myopathy_train+neuropathy_train] = 'neuropathy'
    train_labels = train_labels.decode("utf-8")

    test_data = np.concatenate((emg_healthy[healthy_train:], emg_myopathy[myopathy_train:], emg_neuropathy[neuropathy_train:]))
    test_labels = np.chararray(healthy_test+myopathy_test+neuropathy_test, itemsize=10)
    test_labels[0:healthy_test] = 'healthy'
    test_labels[healthy_test:healthy_test+myopathy_test] = 'myopathy'
    test_labels[healthy_test+myopathy_test:healthy_test+myopathy_test+neuropathy_test] = 'neuropathy'
    test_labels = test_labels.decode("utf-8")
