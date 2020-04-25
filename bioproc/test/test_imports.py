"""
test_imports.py

Author: Praveen Prabhakar KR
Email: praveenp@msu.edu

Module contains the test for imports.
"""

import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
import biosppy as bsp
from biosppy.signals import tools
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from .. import functions as fn
from .. import emg as em

def test_imports():
    """
    Test to check import of libraries.
    """

    #checking scipy, keras, numpy, biosppy
    dummy = sp.__version__
    dummy3 = k.__version__
    dummy4 = np.__version__
    dummy5 = bsp.__version__

    sine = fn.sinewave()
    emg = em.emgsig()

    #checking matplotlib.pyplot
    if plt.plot(sine):
        pass

    #checking tensorflow
    help(tf.keras.Sequential)

    #checking scipy.signal
    if signal.correlate(emg, sine) is not None:
        pass

    #checking biosppy.signals.tools
    if tools.filter_signal(emg, ftype='butter', band='lowpass', order=2,
                           frequency=50, sampling_rate=1000) is not None:
        pass
