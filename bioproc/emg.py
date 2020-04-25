"""
emg.py

Author: Praveen Prabhakar KR
Email: praveenp@msu.edu

Module contains the functions for EMG signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import emg
from . import functions as fn

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

    #plot error
    if plot in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'NO', 'YES']:
        pass
    elif isinstance(plot, str):
        raise ValueError("Plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("Plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #time error
    if tarr in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'NO', 'YES']:
        pass
    elif isinstance(tarr, str):
        raise ValueError("Plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("Plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #checking for random seed
    if seed is not None:
        np.random.seed(seed)

    #simulating EMG signal
    burst1 = np.random.uniform(-1, 1, size=1000) + 0.08
    burst2 = np.random.uniform(-1, 1, size=1000) + 0.08
    quiet = np.random.uniform(-0.05, 0.05, size=500) + 0.08
    emg_signal = np.concatenate([quiet, burst1, quiet, burst2, quiet])

    #plotting
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
        time = np.arange(0, 3.5, 1/1000)
        plt.figure(figsize=(12, 4))
        plt.plot(time, emg_signal)
        plt.title("EMG Simulated Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    if tarr in ['yes', 'Yes', 'Y', 'y', 'YES']:
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

    #signal error
    if isinstance(signal, (list, np.ndarray)):
        signal = np.array(signal)
    elif isinstance(signal, (str, complex)):
        raise TypeError("Signal should be a list or numpy array.")
    else:
        raise ValueError("Signal should be a list or numpy array.")

    for i in signal:
        if isinstance(i, complex):
            raise ValueError("Signal cannot contain complex elements.")

    #sampling frequency error
    try:
        fs = float(fs)
    except TypeError:
        raise TypeError("Sampling frequency (fs) must be int or float.")
    except ValueError:
        raise ValueError("Sampling frequency (fs) must be int or float.")

    #plot error
    if plot in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'NO', 'YES']:
        pass
    elif isinstance(plot, str):
        raise ValueError("Plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("Plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #filt error
    if filt in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'NO', 'YES']:
        pass
    elif isinstance(filt, str):
        raise ValueError("Filt must be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("Filt must be a string - Yes/Y or No/N (non-case sensitive).")

    #filtering
    if filt in ['no', 'No', 'N', 'n', 'NO']:
        signal = fn.iir(signal, plot='No')

    #find onsets
    onsets = emg.find_onsets(signal=signal, sampling_rate=fs, **kwargs)
    onset = np.array(onsets[0])/fs

    #plotting
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
        t = np.arange(0, (len(signal)/fs), 1/fs)
        plt.figure(figsize=(12, 6))
        plt.plot(t, signal)
        xmin, xmax, ymin, ymax = plt.axis()
        plt.vlines(onset, ymin, ymax, colors='#ff7f0e', label="Onsets")
        plt.plot(t, signal, label="EMG signal", c="#1f77b4")
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

    #signal error
    if isinstance(signal, (list, np.ndarray)):
        signal = np.array(signal)
    elif isinstance(signal, (str, complex)):
        raise TypeError("Signal should be a list or numpy array.")
    else:
        raise ValueError("Signal should be a list or numpy array.")

    for i in signal:
        if isinstance(i, complex):
            raise ValueError("Signal cannot contain complex elements.")

    #mvic error
    if isinstance(mvic, (list, np.ndarray)):
        mvic = np.array(mvic)
    elif isinstance(mvic, (str, complex)):
        raise TypeError("MVIC should be a list or numpy array.")
    else:
        raise ValueError("MVIC should be a list or numpy array.")

    #sampling frequency error
    try:
        fs = float(fs)
    except TypeError:
        raise TypeError("Sampling frequency (fs) must be int or float.")
    except ValueError:
        raise ValueError("Sampling frequency (fs) must be int or float.")

    #plot error
    if plot in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'NO', 'YES']:
        pass
    elif isinstance(plot, str):
        raise ValueError("Plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("Plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #filt error
    if filt in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'NO', 'YES']:
        pass
    elif isinstance(filt, str):
        raise ValueError("Filt must be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("Filt must be a string - Yes/Y or No/N (non-case sensitive).")

    #filter and rectify EMG signal
    if filt in ['no', 'No', 'N', 'n', 'NO']:
        signal = fn.iir(signal=signal, fs=fs, plot='No')
        mvic = fn.iir(signal=mvic, fs=fs, plot='No')

    emg_rect = fn.rectify(signal, fs=fs, plot='No')
    mvic_rect = fn.rectify(mvic, fs=fs, plot='No')
    emg_env = fn.envelope(emg_rect, fs=fs, plot='No')
    mvic_env = fn.envelope(mvic_rect, fs=fs, plot='No')
    maximum = mvic_env.max()
    #normalize
    emg_norm = emg_env/maximum

    #plotting
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
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


def padding(signal):
    """
    This function determines the size of the input signal, and zero pads
    such that total size is a power of 2.

    Input parameters
    ----------------
    signal: ndarray
        the input signal

    Output
    ------
    Output will be in the format --> padarray

    padarray: ndarray
        zero padded array; size will be a power of 2
    """

    #signal error
    if isinstance(signal, (list, np.ndarray)):
        signal = np.array(signal)
    elif isinstance(signal, (str, complex)):
        raise TypeError("Signal should be a list or numpy array.")
    else:
        raise ValueError("Signal should be a list or numpy array.")

    for i in signal:
        if isinstance(i, complex):
            raise ValueError("Signal cannot contain complex elements.")

    #pad size and zero padding
    siz = signal.size
    binary = np.binary_repr(siz)
    if binary.count("1") == 1:
        temp = np.log(siz)//np.log(2)
        padarray = fn.padding(signal, size=int(siz))

        return padarray

    else:
        temp = np.log(siz)//np.log(2)
        diff_1 = int((2**(temp+1)) - (2**(temp)))
        diff = int(2**(temp+1)) - siz
        temp_2 = int(2**(temp+1))

        if diff >= diff_1//2:
            padarray = fn.padding(signal, size=int(diff))

            return padarray

        else:
            temp1 = diff+2**(temp+1)
            padarray = fn.padding(signal, size=int(temp1))

            return padarray


def emg_process(emg_signal, mvic_signal=None, fs=1000, plot='Yes', fourier='Yes', **kwargs):
    """
    Function to process emg signal automatically. The function does the following:

    1. Filter
    2. Zero padding
    3. Fast Fourier Transform
    4. Full wave rectification
    5. Envelope
    6. Normalize EMG signal using MVIC

    Input parameters
    ----------------
    emg_signal: ndarray
        input emg signal to be processed
    mvic_signal: ndarray
        maximum voluntary isometric contraction of the same muscle
    fs: int or float, optional
        sampling rate; default set to 1000 Hz
    plot: str - yes/Y or no/N (non-case sensitive), optional
        display the plots or not; default set to Yes
    fourier: str - yes/Y or no/N (non-case sensitive), optional
        plot the FFT or not; default set to Yes
    **kwargs: dict, optional
        Additional keyword arguments as listed:

        ordern: int, optional
            the order of the filter; default set to 2
        cutoff_filter: scalar (int or float) or 2 length sequence (for band-pass, band-stop filter)
            the critical frequency; default set to [50,450]
        ftype: str, optional
            type of filter to be used; default set to 'bandpass'
            types: 'lowpass','highpass', 'bandpass', 'bandstop'
        filter: str, optional
            type of IIR filter; default set to butter
            types: 'butter' (Butterworth), ‘cheby1’ (Chebyshev I), ‘cheby2’ (Chebyshev II),
            ‘ellip’ (Cauer/elliptic), ‘bessel’ (Bessel/Thomson)
        order_env: int, optional
            the order of the filter for envelope; default set to 2
        cutoff_env: int or float, optional
            the critical frequency for envelope; default set to 10 Hz
        filter_env: str, optional
            type of IIR filter for envelope; default set to butter
            types: 'butter' (Butterworth), ‘cheby1’ (Chebyshev I), ‘cheby2’ (Chebyshev II),
            ‘ellip’ (Cauer/elliptic), ‘bessel’ (Bessel/Thomson)

    Output
    ------
    Output will be in the format --> emgnorm (or) env

    emgnorm: ndarray
        MVIC normalized EMG signal
    env: ndarray
        linear envelope of the input signal
    """

    #EMG signal error
    if isinstance(emg_signal, (list, np.ndarray)):
        emg_signal = np.array(emg_signal)
    elif isinstance(emg_signal, (str, complex)):
        raise TypeError("EMG signal should be a list or numpy array.")
    else:
        raise ValueError("EMG signal should be a list or numpy array.")

    for i in emg_signal:
        if isinstance(i, complex):
            raise ValueError("EMG signal cannot contain complex elements.")

    #MVIC signal error
    if mvic_signal is not None:
        if isinstance(mvic_signal, (list, np.ndarray)):
            mvic_signal = np.array(mvic_signal)
        elif isinstance(mvic_signal, (str, complex)):
            raise TypeError("MVIC signal should be a list or numpy array.")
        else:
            raise ValueError("MVIC signal should be a list or numpy array.")

        for i in mvic_signal:
            if isinstance(i, complex):
                raise ValueError("MVIC signal cannot contain complex elements.")

    #kwargs
    list1 = ['ordern', 'cutoff_filter', 'ftype', 'filter', 'order_env', 'cutoff_env', 'filter_env']
    keyl = []
    assign = []

    for key, values in kwargs.items():
        keyl.append(key)

    for i in list1:
        if i in list1:
            if i in keyl:
                assign.append(i)

    if 'ordern' in assign:
        ordern = kwargs['ordern']
    else:
        ordern = 2

    if 'cutoff_filter' in assign:
        cutoff_filter = kwargs['cutoff_filter']
    else:
        cutoff_filter = [50, 450]

    if 'ftype' in assign:
        ftype = kwargs['ftype']
    else:
        ftype = 'bandpass'

    if 'filter' in assign:
        filter = kwargs['filter']
    else:
        filter = 'butter'

    if 'order_env' in assign:
        order_env = kwargs['order_env']
    else:
        order_env = 2

    if 'cutoff_env' in assign:
        cutoff_env = kwargs['cutoff_env']
    else:
        cutoff_env = 10

    if 'filter_env' in assign:
        filter_env = kwargs['filter_env']
    else:
        filter_env = 'butter'

    #processing
    filtered = fn.iir(signal=emg_signal, fs=fs, ordern=ordern, cutoff=cutoff_filter,
                      ftype=ftype, filter=filter, plot='No')
    rect = fn.rectify(filtered, fs=fs, plot='No')
    env = fn.envelope(rect, fs=fs, order=order_env, cutoff=cutoff_env, filter=filter_env, plot='N')

    if mvic_signal is None:
        print("MVIC signal not input. Cannot normalize EMG signal.")
    else:
        emgnorm = norm_emg(signal=emg_signal, mvic=mvic_signal, fs=fs, filt='No', plot='No')

    if mvic_signal is None:
        if plot in ['Yes', 'yes', 'Y', 'y', 'YES']:
            plt.figure(figsize=(12, 12))
            time = np.arange(0, emg_signal.size/fs, 1/fs)
            plt.subplot(411)
            plt.plot(time, emg_signal)
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("Raw EMG signal")

            plt.subplot(412)
            plt.plot(time, filtered, c='#ff7f0e')
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("Filtered signal")

            plt.subplot(413)
            plt.plot(time, rect)
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("Rectified signal")

            plt.subplot(414)
            plt.plot(time, env, c='#ff7f0e')
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("Envelope")
            plt.tight_layout(pad=3.0)
            plt.show()

            if fourier in ['Yes', 'yes', 'Y', 'y', 'YES']:
                padded = padding(filtered)
                dft = fn.fft(padded, fs=fs, plot='Y')

        return env

    else:
        if plot in ['Yes', 'yes', 'Y', 'y', 'YES']:
            plt.figure(figsize=(12, 18))
            time = np.arange(0, emg_signal.size/fs, 1/fs)
            plt.subplot(511)
            plt.plot(time, emg_signal)
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("Raw EMG signal")

            plt.subplot(512)
            plt.plot(time, filtered, c='#ff7f0e')
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("Filtered EMG")

            plt.subplot(513)
            plt.plot(time, rect)
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("Rectified EMG")

            plt.subplot(514)
            plt.plot(time, env, c='#ff7f0e')
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("EMG Envelope")

            plt.subplot(515)
            plt.plot(time, emgnorm, c='#ff7f0e')
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("Normalized EMG")
            plt.tight_layout(pad=3.0)
            plt.show()

            if fourier in ['Yes', 'yes', 'Y', 'y', 'YES']:
                padded = padding(filtered)
                fn.fft(padded, fs=fs, plot='Y')

        return emgnorm
