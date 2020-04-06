import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import biosppy as bsp
from biosppy.signals import tools
    
def sinewave(amp=1, freq=1, time=10, fs=100, phi=0, offset=0, plot='yes', tarr='no'):
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
    plot: str - yes or no, optional
        plot the sine wave or not; default set to yes
    tarr: str - yes or no, optional
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
    
def sinenoise(amp=1, freq=1, time=10, fs=100, phi=0, offset=0, noise=1, plot='yes', tarr='no'):
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
    plot: str - yes or no, optional
        plot the sine wave or not; default set to yes
    tarr: str - yes or no, optional
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


def emgsig(seed=None, plot='Yes'):
    """
    
    Simulate an EMG signal for time = 3.5 secs.
    
    Input parameters
    ----------------
    seed: int, optional
        initialize seed of random number generator
    plot: str - yes or no, optional
        plot the EMG wave or not; default set to yes
        
    Output
    ------
    Output will be in the format --> emg
    
    t: ndarray
        simulated emg signal
        
    """
    
    if seed is not None:
        np.random.seed(seed)
        
    burst1 = np.random.uniform(-1, 1, size=1000) + 0.08
    burst2 = np.random.uniform(-1, 1, size=1000) + 0.08
    quiet = np.random.uniform(-0.05, 0.05, size=500) + 0.08
    emg = np.concatenate([quiet, burst1, quiet, burst2, quiet])
    
    if plot == 'yes' or plot == 'Yes':
        t = np.arange(0,3.5,1/1000)
        plt.plot(t,emg)
        plt.title("EMG Simulated Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (m)")
    
    return emg


def fft(sinwave, fs=1000, n=None, axis= -1, plot='yes'):
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
    plot: str - yes or no, optional
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
        

def padding(signal, size=0):
    """
    
    This function adds zero padding to a signal. 
    
    Input parameters
    ----------------
    signal: ndarray
        the input signal for zero padding
    size: int, optional
        number of zeros to be added to the signal
    
    Output
    ------
    Output will be in the format --> padarray
    
    padarray: ndarray
        zero padded signal
    
    """
    
    padarray = np.concatenate((signal, np.zeros(size)))
    
    
    return padarray


def padsize(signal, retarr='Yes'):
    """
    
    This function determines the size of the input signal, suggests sizes for zero padding such that total size is a power of 2.
    
    Input parameters
    ----------------
    signal: ndarray
        the input signal 
    retarr: yes or no, optional
        return a zero padded signal array or not; default set to yes
    
    Output
    ------
    Output will be in the format --> padarray
    
    padarray: ndarray
        zero padded array; size will be a power of 2
    
    """
    
    siz = signal.size
    bin = np.binary_repr(siz)
    if bin.count("1") == 1:
        temp = np.log(siz)//np.log(2)
        print("""The size of the signal is {}, which is a power of 2 (2^{} = {}. Suggestion: Add {} more zeros using the padding function"""
              .format(siz,int(temp), int(2**(temp)), siz))
        inp = input("Do you want to add {} more zeros using the padding function? Y/N".format(siz))
        if inp == 'Y' or inp == 'y':
            if retarr == 'Yes' or retarr == 'yes':
                return padding(signal,size = int(siz))
        
    else:
        temp = np.log(siz)//np.log(2)
        diff = (2**(temp+1)) - siz
        
        print("""The size of the signal is {}. The closest power of 2 is 2^{} = {}. Suggestion: Add {} more zeros to bring it to closest power of 2. Better solution is to add {} zeros."""
              .format(siz, int(temp+1), int(2**(temp+1)), diff, diff+2**(temp+1)))
        inp = input("Do you want to add {} more zeros using the padding function? Y/N".format(diff+2**(temp+1)))
        if inp == 'Y' or inp == 'y':
            if retarr == 'Yes' or retarr == 'yes':
                return padding(signal,size = int(diff+2**(temp+1))) 


def window(wave):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return nwave


def iir(signal, fs=1000, ordern=2, cutoff=[50,500], ftype='bandpass', filter='butter', ripple='None', att='None', plot='Yes' ):
    """
    This function applies a digital IIR filter to the input signal and returns the filtered signal.
    
    Input parameters
    ----------------
    signal: ndarray
        the input signal to be filter
    fs: int or float, optional
        sampling rate
    ordern: int, optional
        the order of the filter; default set to 2
    cutoff: scalar (int or float) or 2 length sequence (for band-pass and band-stop filter)
        the critical frequency; default set to [50,500]
    ftype: str, optional
        type of filter to be used; default set to 'bandpass'
        types: 'lowpass','highpass', 'bandpass', 'bandstop' 
    filter: str, optional
        type of IIR filter; default set to butter
        types: 'butter' (Butterworth), ‘cheby1’ (Chebyshev I), ‘cheby2’ (Chebyshev II), 
        ‘ellip’ (Cauer/elliptic), ‘bessel’ (Bessel/Thomson) 
    ripple: float, optional
        maximum ripple in the passband (for Chebyshev and Elliptical filters); default set to None
    att: float, optional
        minimum attenuation in the stop band (for Chebyshev and Elliptical filters); default set to None
    plot:  str - yes or no, optional
        plot the filtered signal or not; default set to yes
    
    Output
    ------
    Output will be in the format --> filtersig
    
    filtersig: ndarray
        the filtered signal
    
    """
    try:
        cutoff = float(cutoff)
    except:
        cutoff = np.array(cutoff)
    
    
    filtersig = tools.filter_signal(signal, ftype=filter, band=ftype, order=ordern, frequency=cutoff, sampling_rate=fs)
    filtersig = filtersig['signal']
    
    if plot=='Yes' or plot=='yes':
        time = np.arange(0,len(signal)/fs, 1/fs)
        plt.plot(time, filtersig)
    
    return filtersig


def fir(signal, ordern=2, cutoff=[50,500], ftype='bandpass', fs=1000.0, plot='Yes'):
    """
    Apply a FIR filter to the input signal.
    
    Input parameters
    ----------------
    signal: ndarray
        the input signal to be filter
    order: int, optional
        the order of the filter; default set to 2
    cutoff: scalar (int or float) or 2 length sequence (for band-pass and band-stop filter)
        the critical frequency; default set to [50,500]
    ftype: str, optional
        type of filter to be used; default set to 'bandpass'
        types: 'lowpass','highpass', 'bandpass', 'bandstop' 
    fs: int or float, optional
        sampling rate
    plot: str - yes or no, optional
        plot the filtered signal or not; default set to yes
    
    Output
    ------
    Output will be in the format --> filtersig
    
    filtersig: ndarray
        the filtered signal
    
    """
    try:
        cutoff = float(cutoff)
    except:
        cutoff = np.array(cutoff)
    
    cutoff = (2*cutoff)/fs
    filtersig = tools.filter_signal(signal, ftype='FIR', band=ftype, order=ordern, frequency=cutoff, sampling_rate=fs)
    filtersig = filtersig['signal']

    if plot=='Yes' or plot=='yes':
        time = np.arange(0,len(signal)/fs, 1/fs)
        plt.plot(time, filtersig)
    
    return filtersig


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

def ccorr(sig1, sig2):
    """
    Calculate the cross correlation of two signals.
    
    Input parameters
    ----------------
    sig1: ndarray
        first signal for cross correlation
    sig2: ndarray
        second signal for cross correlation
    
    Output
    ------
    Output will be in the format --> lag, corrarray
    
    lag: ndarray
        lag indices
    corrarray: ndarray
        cross correlation
    
    """
    
    corr = plt.xcorr(sig1,sig2, maxlags=None)
    corr = np.array(corr)
    lag = corr[0,]
    corrarray = corr[1,]
    
    return lag, corrarray


def acorr(signal):
    """
    Calculate the auto-correlation of a signal.
    
    Input parameters
    ----------------
    signal: ndarray
        signal whose auto-correlation is to be calculated
    
    Output
    ------
    Output will be in the format --> lag, acorrarray
    
    lag: ndarray
        lag indices
    acorrarray: ndarray
        auto correlation
    
    """
    
    corr = plt.xcorr(signal,signal, maxlags=None)
    corr = np.array(corr)
    lag = corr[0,]
    acorrarray = corr[1,]
    
    return lag, acorrarray


def psd(array):
    """
    This function 
    
    Input parameters
    ----------------
    
    
    Output
    ------
    
    """
    
    
    return narray


def rectify(signal, fs=1000, plot='Yes'):
    """
    Full wave rectification of the input signal.
    
    Input parameters
    ----------------
    signal: ndarray
        input signal
    fs: int or float, optional
        sampling frequency; set to 1000 Hz by default
    plt: str - yes or no, optional
        plot the rectified signal or not; default set to yes
    
    Output
    ------
    Output will be in the format --> rectifiedsig
    
    rectifiedsig: ndarray
        rectified signal array
    
    """
    
    rectifiedsig = np.abs(signal)
    
    if plot == 'Yes' or plot == 'yes':
        time = np.arange(0,len(signal)/fs, 1/fs)
        plt.plot(time, rectifiedsig)
    
    return rectifiedsig

def envelope(signal, fs=1000, order=2, cutoff=10, filter='butter', plot='Yes'):
    """
    Linear envelope of the input signal, computed by low pass fitering the rectified signal.
    
    Input parameters
    ----------------
    signal: ndarray
        input signal
    fs: int or float, optional
        sampling rate; default set to 1000 Hz
    order: int, optional
        the order of the filter; default set to 2
    cutoff: int or float, optional
        the critical frequency; default set to 20 Hz 
    filter: str, optional
        type of IIR filter; default set to butter
        types: 'butter' (Butterworth), ‘cheby1’ (Chebyshev I), ‘cheby2’ (Chebyshev II), 
        ‘ellip’ (Cauer/elliptic), ‘bessel’ (Bessel/Thomson) 
    plot:  str - yes or no, optional
        plot the linear envelope or not; default set to yes
    
    Output
    ------
    Output will be in the format --> linenv
    
    linenv: ndarray
        linear envelope of the input signal
    
    """
    
    temp = rectify(signal, plot='No')
    linenv = iir(temp, fs=fs, ordern=order, cutoff=cutoff/2, ftype='lowpass', filter='butter', ripple='None', att='None', plot='No' )
    
    if plot == 'Yes' or plot == 'yes':
        time = np.arange(0,len(signal)/fs, 1/fs)
        plt.plot(time, linenv)
    
    return linenv


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
