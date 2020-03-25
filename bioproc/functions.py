
import numpy as np
import scipy
from scipy.fftpack import fft
import matplotlib.pyplot as plt
%matplotlib inline 
#for Jupyter notebook

    
def sinewave(amp=1, freq=1, time = 10, fs=100, phi=0, offset=0):
    """ 
    This function is used to generate sine waves without any noise.
    
    Input parameters
    ----------------
    amp: int or float
        amplitude of the sine wave; set to 1 by default
    freq: int or float 
        frequency of the sine wave; set to 1 Hz by default
    time: int or float
        time period of the sine wave; set to 10 secs by default
    fs: int or float
        sampling frequency; set to 1000 Hz by default
        
    Output
    ------
    Output will be in the format --> t,sine
    
    t: int or float
       time of sine wave in secs
    sine: sine wave
        amplitude of sine wave
    
    """ 
    
    pi = 3.14
    t = np.arange(0,time, 1/fs)
    sine = offset + np.sin((2*pi*freq*t) + phi)
    
    return t,sine
    
    
def sinenoise(amp=1, freq=1, time = 10, fs=100, phi=0, offset=0):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    pi = 3.14
    t = np.arange(0,time, 1/fs)
    sine = offset + np.sin((2*pi*freq*t) + phi) + (np.random.randn(len(t))/10)
    
    return sine


def fft(sin, n=None, axis= -1, o=False):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return fourier


def padding(wave):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return nwave


def window(wave):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return nwave


def iir(wave):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return nwave


def fir(wave):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return nwave

def movavg(wave):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return nwave


def polyfit(wave):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return nwave

def splinefit(wave):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return nwave

def fourierfit(wave):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return nwave

def ccorr(array):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return corrarray


def acorr(array):
"""
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    

    return acorrarray

