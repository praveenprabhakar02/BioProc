"""
EMG signal processing.
"""

import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
from biosppy.signals import emg
from . import functions as fn
#import biosppy as bsp
#from biosppy.signals import tools

def emgsig(seed=None, plot='no', tarr='no'):
    """
    Simulate an EMG signal for time = 3.5 secs.

    Input parameters
    ----------------
    seed: int, optional
        initialize seed of random number generator
    plot: str - yes or no, optional
        plot the EMG wave or not; default set to no
    tarr: str - yes or no, optional
        return the time array or not; default set to no

    Output
    ------
    Output will be in the format --> t,emg

    time: ndarray
       time array
    emg: ndarray
       simulated emg signal
    """

    #checking for random seed
    if seed is not None:
        np.random.seed(seed)

    #simulating EMG signal
    burst1 = np.random.uniform(-1, 1, size=1000) + 0.08
    burst2 = np.random.uniform(-1, 1, size=1000) + 0.08
    quiet = np.random.uniform(-0.05, 0.05, size=500) + 0.08
    emg_signal = np.concatenate([quiet, burst1, quiet, burst2, quiet])

    #plotting
    if plot in ['yes', 'Yes']:
        time = np.arange(0, 3.5, 1/1000)
        plt.figure(figsize=(12, 4))
        plt.plot(time, emg_signal)
        plt.title("EMG Simulated Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    if tarr in ['yes', 'Yes']:
        return time, emg_signal

    return emg_signal


def find_onset(signal, fs=1000.0, filt='No', plot='Yes', **kwargs):
    """
    Find onset of EMG signals.

    Input parameters
    ----------------
    signal: ndarray
        filtered input signal
    fs: int or float, optional
        sampling rate; default set to 1000 Hz
    filt: str - yes or no, optional
        whether the input signal is filtered or not; default set to No
    plot: str - yes or no, optional
        plot the onsets or not; default set to yes
    **kwargs : dict, optional
        Additional keyword arguments are passed to the underlying
        biosppy.signals.emg.find_onsets function.

    Output
    ------
    Output will be in the format --> onsets

    onsets: ndarray
        the onset of the EMG signal
    """

    if filt in ['no', 'No']:
        signal = fn.iir(signal, plot='No')

    onsets = emg.find_onsets(signal=signal, sampling_rate=fs, **kwargs)
    onset = np.array(onsets[0])/fs

    #plotting
    if plot in ['yes', 'Yes']:
        t = np.arange(0, (len(signal)/fs), 1/fs)
        for i in onset:
            plt.axvline(i, c='#ff7f0e', label="Onsets")
        plt.plot(t, signal, label="EMG signal")
        plt.title("EMG onsets")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    return onset


def norm_emg(signal, mvic, fs=1000.0, filt='no', plot='yes'):
    """
    Find onset of EMG signals.

    Input parameters
    ----------------
    signal: ndarray
        input EMG signal of a muscle
    mvic: ndarray
        maximum voluntary isometric contraction of the same muscle
    fs: int or float, optional
        sampling rate; default set to 1000 Hz
    filt: str - yes or no, optional
        whether the input signal and mvic are filtered or not; default set to No
    plot: str - yes or no, optional
        plot the onsets or not; default set to Yes

    Output
    ------
    Output will be in the format --> emg_norm

    emg_norm: ndarray
        MVIC normalized EMG signal
    """

    #filter and rectify EMG signal
    if filt in ['no', 'No']:
        emg = fn.iir(signal=signal, fs=fs, plot='No')
        emg_rect = fn.rectify(emg, fs=fs, plot='No')
        mvic = fn.iir(signal=mvic, fs=fs, plot='No')
        mvic_rect = fn.rectify(mvic, fs=fs, plot='No')

    maximum = mvic_rect.max()
    #normalize
    emg_norm = emg_rect/maximum

    #plotting
    if plot in ['yes', 'Yes']:
        t = np.arange(0, (len(signal)/fs), 1/fs)

        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        plt.suptitle("Rectified EMG & Normalized EMG")
        plt.plot(t, emg_rect, label="Rectified EMG")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.subplot(212)
        plt.plot(t, emg_norm, c='#ff7f0e', label="Normalized EMG")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    return emg_norm


#class emgs():
#    """
#    """
