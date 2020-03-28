
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
%matplotlib inline 
#for Jupyter notebook

    
def sinewave(amp=1, freq=1, time = 10, fs=100, phi=0, offset=0, plot='yes', tarr='no'):
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
        sampling frequency; set to 100 Hz by default
    phi: int or float
        phase (angle); set to 0 degrees by default
    offset: int or float
        bias; set to 1 by default
    plot: string - yes or no
        Plot the sine wave or not; default set to yes
    tarr: string - yes or no
        return the time array or not; default set to no
        
    Output
    ------
    Output will be in the format --> t,sine
    
    t: int or float
       time of sine wave in secs
    sine: sine wave
        amplitude of the sine wave
    
    """ 
    
    pi = 3.14
    t = np.arange(0,time, 1/fs)
    sine = offset + np.sin((2*pi*freq*t) + phi)
    
    if plot == 'yes':
        plt.plot(t,sine)
        plt.title("Sine wave")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (m)")
    
    if tarr == 'yes':
        return t,sine
    else:
        return sine
    
def sinenoise(amp=1, freq=1, time = 10, fs=100, phi=0, offset=0, noise = 1, plot='yes', tarr='no'):
    """
    
    This function is used to generate a sine wave with random noise.
    
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
    phi: int or float
        phase (angle); set to 0 degrees by default
    offset: int or float
        bias; set to 1 by default
    noise: int or float
        control the level of noise; increasing the value reduces the noise
        and vice versa
    plot: string - yes or no
        Plot the sine wave or not; default set to yes
    tarr: string - yes or no
        return the time array or not; default set to no 
        
    Output
    ------
    Output will be in the format --> t,sine
    
    t: int or float
       time of sine wave in secs
    sine: sine wave
        amplitude of the sine wave
        
    """
    
    pi = 3.14
    t = np.arange(0,time, 1/fs)
    sine = offset + np.sin((2*pi*freq*t) + phi) + (np.random.randn(len(t))/noise)
    
    if plot == 'yes':
        plt.plot(t,sine)
        plt.title("Sine wave with noise")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (m)")
        
    if tarr == 'yes':
        return t,sine
    else:
        return sine


def fft(sinwave, fs = 1000, n=None, axis= -1, plot='yes'):
    """
    
    This function is used to calculate the discrete fourier transform of the sine wave.
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    
    """
    n = axis
    fourier = np.fft.fft(sinwave, axis = n)
    N = sinwave.size
    amp = np.linspace(0, fs, N)
    
    if plot == 'yes':
        
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency (Hz)")
        plt.bar(amp[:N // 2], np.abs(fourier)[:N // 2], width=1.5)
        plt.show

        return fourier
    
    else:
        
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


def psd(array):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return narray


def rectify(array):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return narray

def integrate(array):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return narray


def rms(array):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return narray
