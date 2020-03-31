import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

    
def sinewave(amp=1, freq=1, time = 10, fs=100, phi=0, offset=0, plot='yes', tarr='no'):
    """ 
    
    This function is used to generate sine waves without any noise.
    
    Input parameters
    ----------------
    amp: int or float, optional
        amplitude of the sine wave; set to 1 by default
    freq: int or float, optional 
        frequency of the sine wave; set to 1 Hz by default
    time: int or float, optional
        time period of the sine wave; set to 10 secs by default
    fs: int or float, optional
        sampling frequency; set to 100 Hz by default
    phi: int or float, optional
        phase (angle); set to 0 degrees by default
    offset: int or float, optional
        bias; set to 1 by default
    plot: string - yes or no, optional
        plot the sine wave or not; default set to yes
    tarr: string - yes or no, optional
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
    
    if plot == 'yes' or plot == 'Yes':
        plt.plot(t,sine)
        plt.title("Sine wave")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (m)")
    
    if tarr == 'yes' or tarr == 'Yes':
        return t,sine
    elif tarr == 'no' or tarr == 'No':
        return sine
    
def sinenoise(amp=1, freq=1, time = 10, fs=100, phi=0, offset=0, noise = 1, plot='yes', tarr='no'):
    """
    
    This function is used to generate a sine wave with random noise.
    
    Input parameters
    ----------------
    amp: int or float, optional
        amplitude of the sine wave; set to 1 by default
    freq: int or float, optional
        frequency of the sine wave; set to 1 Hz by default
    time: int or float, optional
        time period of the sine wave; set to 10 secs by default
    fs: int or float, optional
        sampling frequency; set to 1000 Hz by default
    phi: int or float, optional
        phase (angle); set to 0 degrees by default
    offset: int or float, optional
        bias; set to 1 by default
    noise: int or float, optional
        control the level of noise; increasing the value reduces the noise and vice versa
    plot: string - yes or no, optional
        plot the sine wave or not; default set to yes
    tarr: string - yes or no, optional
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
    
    if plot == 'yes' or plot == 'Yes':
        plt.plot(t,sine)
        plt.title("Sine wave with noise")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (m)")
        
    if tarr == 'yes' or tarr == 'Yes':
        return t,sine
    elif tarr == 'no' or tarr == 'No':
        return sine


def fft(sinwave, fs = 1000, n=None, axis= -1, plot='yes'):
    """
    
    This function is used to calculate the discrete fourier transform of the sine wave.
    
    Input parameters
    ----------------
    sinwave: ndarray
        the input signal whose DFT is to be determined
    fs: int or float, optional
        sampling frequency; default set to 1000 Hz
    n: int, optional
        length of the transformed axis of the output; default set to None
    axis: int, optional
        axis over which to compute FFT; default set to -1
    plot: string - yes or no, optional
        plot the fourier or not; default set to yes
    
    
    Output
    ------
    Output will be in the format --> fourier
    
    fourier: ndarray
        FFT of the input signal
    
    """
    n = axis
    fourier = np.fft.fft(sinwave, axis = n)
    N = sinwave.size
    amp = np.linspace(0, fs, N)
    
    if plot == 'yes' or plot == 'Yes':
        
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency (Hz)")
        plt.bar(amp[:N // 2], np.abs(fourier)[:N // 2]*1/N, width=1.5)
        plt.show

        return fourier
    
    elif plot == 'no' or plot == 'No':
        
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

def movavg(signal, n=3):
    """
    This function returns the moving average of a signal.
    
    Input parameters
    ----------------
    signal: ndarray
        the input signal whose moving average is to be determined
    n: int, optional
        the number of points used for moving average
    
    Output
    ------
    Output will be in the format --> moveaverage
    
    movaverage: ndarray
        the moving average of the signal    
    
    """
    count = 0
    temp = signal.size
    movaverage = []

    while((count + n) <= temp):

        temp1 = np.sum(signal[count:(count+n)]) / n
        movaverage.append(temp1)
        count += 1

    movaverage = np.array(movaverage)
    
    return movaverage


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


def rms(signal):
    """
    This function returns the Root Mean Square of the signal.
    
    Input parameters
    ----------------
    signal: ndarray
        the input signal whose RMS is to be determined
    
    Output
    ------
    Output will be in the format --> rmsq
    
    rmsq: int or float
        the root mean square of the signal
    
    """
    
    rmsq = np.sqrt(np.sum(signal**2)/signal.size)
    
    return rmsq
