"""
Contains the functions for processing signals.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#import biosppy as bsp
from biosppy.signals import tools
from biosppy.signals import emg

def sinewave(amp=1, freq=1, time=10, fs=100, phi=0, offset=0, plot='no', tarr='no'):
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
        plot the sine wave or not; default set to no
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
    t = np.arange(0, time, 1/fs)
    sine = offset + amp * (np.sin((2*pi*freq*t) + phi))

    #plotting
    if plot in ['yes', 'Yes']:
        plt.figure(figsize=(12, 4))
        plt.plot(t, sine)
        plt.title("Sine wave")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    #return time array or not
    if tarr in ['yes', 'Yes']:
        return t, sine

    return sine


def sinenoise(amp=1, freq=1, time=10, fs=100, phi=0, offset=0, noise=1, plot='no', tarr='no'):
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
        noise amplitude; default set to 1
    plot: str - yes or no, optional
        plot the sine wave or not; default set to no
    tarr: str - yes or no, optional
        return the time array or not; default set to no

    Output
    ------
    Output will be in the format --> t,sine

    t: ndarray
       time array
    sine: sine wave
        amplitude of the sine wave
    """

    pi = 3.14
    t = np.arange(0, time, 1/fs)
    sine = offset + amp * (np.sin((2*pi*freq*t) + phi)) + (noise *(np.random.randn(len(t))))

    #plotting
    if plot in ['yes', 'Yes']:
        plt.figure(figsize=(12, 4))
        plt.plot(t, sine)
        plt.title("Sine wave with noise")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (m)")
        plt.show()

    #return time array or not
    if tarr in ['yes', 'Yes']:
        return t, sine

    return sine


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


def fft(sinwave, fs=1000, plot='yes', **kwargs):
    """
    This function is used to calculate the discrete fourier transform of the sine wave.

    Input parameters
    ----------------
    sinwave: ndarray
        the input signal whose DFT is to be determined
    fs: int or float, optional
        sampling frequency; default set to 1000 Hz
    plot: str - yes or no, optional
        plot the fourier or not; default set to yes
    **kwargs: dict, optional
        Additional keyword arguments are passed to the underlying
        numpy.fft.fft function

    Output
    ------
    Output will be in the format --> fourier

    fourier: ndarray
        FFT of the input signal
    """

    fourier = np.fft.fft(sinwave, **kwargs)
    N = sinwave.size
    amp = np.linspace(0, fs, N)

    #plotting
    if plot in ['yes', 'Yes']:
        plt.title("FFT")
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency")
        plt.bar(amp[:N // 2], np.abs(fourier)[:N // 2]*1/N, width=1.5)
        plt.show()

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


def padsize(signal):
    """
    This function determines the size of the input signal, suggests
    sizes for zero padding such that total size is a power of 2.

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

    siz = signal.size
    binary = np.binary_repr(siz)
    if binary.count("1") == 1:
        temp = np.log(siz)//np.log(2)
        temp_2 = int(2**int(temp))
        print("""The size of the signal is {}, which is a power of 2 (2^{})={}.
        \nSuggestion:\nAdd {} more zeros using the padding function"""
              .format(siz, temp, temp_2, siz))
        inp = input("\nDo you want to add {} more zeros using padding function? Y/N".format(siz))
        if inp in ['y', 'Y', 'yes', 'Yes']:
            return padding(signal, size=int(siz))

    else:
        temp = np.log(siz)//np.log(2)
        diff_1 = int((2**(temp+1)) - (2**(temp)))
        diff = int(2**(temp+1)) - siz
        temp_2 = int(2**(temp+1))

        if diff >= diff_1//2:
            print("""The size of the signal is {}. The closest power of 2 is 2^{}={}.
            \nSuggestion:\nAdd {} more zeros to bring it to closest power of 2."""
                  .format(siz, temp+1, temp_2, diff))
            inp = input("\nDo you want to add {} more zeros using padding function? Y/N"
                        .format(diff))
            if inp in ['y', 'Y', 'yes', 'Yes']:
                return padding(signal, size=int(diff))

        else:
            temp1 = diff+2**(temp+1)
            print("""The size of the signal is {}. The closest power of 2 is 2^{}={}.
            \nSuggestion:\nAdd {} more zeros to bring it to {}. Better solution: add {} zeros."""
                  .format(siz, int(temp+1), int(2**(temp+1)), int(2**(temp+1)), diff, temp1))
            inp = input("\nDo you want to add {} more zeros using the padding function? Y/N"
                        .format(temp1))
            if inp in ['y', 'Y', 'yes', 'Yes']:
                return padding(signal, size=int(temp1))


def window(kernel, size, **kwargs):
    """
    Return a window for the given parameters.

    Input parameters
    ----------------
    kernel : str
        Type of window to create.
    size : int
        Size of the window.
    **kwargs : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal.windows function

    Output
    ------
    Output will be in the format --> windows

    windows : ndarray
        Created window.
    """

    windows = tools._get_window(kernel, size, **kwargs)

    return windows


def iir(signal, fs=1000, ordern=2, cutoff=[50, 450], ftype='bandpass', filter='butter',
        plot='yes', **kwargs):
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
        the critical frequency; default set to [50,450]
    ftype: str, optional
        type of filter to be used; default set to 'bandpass'
        types: 'lowpass','highpass', 'bandpass', 'bandstop'
    filter: str, optional
        type of IIR filter; default set to butter
        types: 'butter' (Butterworth), ‘cheby1’ (Chebyshev I), ‘cheby2’ (Chebyshev II),
        ‘ellip’ (Cauer/elliptic), ‘bessel’ (Bessel/Thomson)
    plot:  str - yes or no, optional
        plot the filtered signal or not; default set to yes
    **kwargs: dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal function

    Output
    ------
    Output will be in the format --> filtersig

    filtersig: ndarray
        the filtered signal
    """

    try:
        cutoff = float(cutoff)
    except TypeError:
        cutoff = np.array(cutoff)
    except ValueError:
        raise ValueError("Cutoff can only be an int, float or numpy array")

    #filtering
    filtersig = tools.filter_signal(signal, ftype=filter, band=ftype, order=ordern,
                                    frequency=cutoff, sampling_rate=fs, **kwargs)
    filtersig = filtersig['signal']

    #plotting
    if plot in ['yes', 'Yes']:
        time = np.arange(0, len(signal)/fs, 1/fs)
        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        plt.plot(time, signal, label="Raw signal")
        plt.suptitle("Raw & Filtered signal")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.subplot(212)
        plt.plot(time, filtersig, c='#ff7f0e', label="Filtered signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    return filtersig


def fir(signal, ordern=2, cutoff=[50, 450], ftype='bandpass', fs=1000.0, plot='yes', **kwargs):
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
    **kwargs: dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal function

    Output
    ------
    Output will be in the format --> filtersig

    filtersig: ndarray
        the filtered signal
    """

    try:
        cutoff = float(cutoff)
    except TypeError:
        cutoff = np.array(cutoff)
    except ValueError:
        raise ValueError("Cutoff can only be an int, float or numpy array")

    #filtering
    filtersig = tools.filter_signal(signal, ftype='FIR', band=ftype, order=ordern,
                                    frequency=cutoff, sampling_rate=fs, **kwargs)
    filtersig = filtersig['signal']

    #plotting
    if plot in ['yes', 'Yes']:
        time = np.arange(0, len(signal)/fs, 1/fs)
        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        plt.plot(time, signal, label="Raw signal")
        plt.suptitle("Raw & Filtered signal")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.subplot(212)
        plt.plot(time, filtersig, c='#ff7f0e', label="Filtered signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    return filtersig


def movavg(signal, window_size=3):
    """
    This function returns the moving average of a signal.

    Input parameters
    ----------------
    signal: ndarray
        the input signal whose moving average is to be determined
    window_size: int, optional
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

    while (count + window_size) <= temp:

        temp1 = np.sum(signal[count:(count + window_size)]) / window_size
        movaverage.append(temp1)
        count += 1

    movaverage = np.array(movaverage)

    return movaverage


def polyfit(time, signal, degree, plot='yes', **kwargs):
    """
    Polynomial regression.

    Input parameters
    ----------------
    time: ndarray
        independent variable
    signal: ndarray
        dependent variable
    plot: str - yes or no, optional
        plot the polyfit or not; default set to yes
    **kwargs: dict, optional
        Additional keyword arguments are passed to the underlying
        np.polyfit function

    Output
    ------
    Output will be in the format --> regression

    regression: ndarray
        polynomial regression array
    """

    parameters = np.polyfit(time, signal, degree, **kwargs)
    poly_function = np.poly1d(parameters)
    regression = poly_function(time)

    #plotting
    if plot in ['yes', 'Yes']:
        plt.plot(time, signal, label="Signal")
        plt.plot(time, regression, c='#ff7f0e', label="Polynomial")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Polynomial fitting")
        plt.legend()
        plt.show()

    return regression


def splinefit(time, signal, res=2000, plot='yes', **kwargs):
    """
    One dimensional smoothing spline fit.

    Input parameters
    ----------------
    time: ndarray
        time array, independent variable
    signal: ndarray
        signal array, dependent variable
    res: int, optional
        resolution, number of points; default set to 2000
    plot: str - yes or no, optional
        plot the spline or not; default set to yes
    **kwargs : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.interpolate.UnivariateSpline function

    Output
    ------
    Output will be in the format --> times, spline(times)

    times: ndarray
        time array
    splines_final: ndarray
        spline fitted array
    """

    spline = sp.interpolate.UnivariateSpline(time, signal, **kwargs)
    times = np.linspace(np.min(time), np.max(time), res)
    spline.set_smoothing_factor(0.5)
    splines_final = spline(times)

    #plotting
    if plot in ['yes', 'Yes']:
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal, label="Signal")
        plt.plot(times, splines_final, lw=3, c='#ff7f0e', label="Spline")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Polynomial fitting")
        plt.legend()
        plt.show()

    return times, splines_final


def ccorr(sig1, sig2, **kwargs):
    """
    Calculate the cross correlation of two signals.

    Input parameters
    ----------------
    sig1: ndarray
        first signal for cross correlation
    sig2: ndarray
        second signal for cross correlation
    **kwargs : dict, optional
        Additional keyword arguments are passed to the underlying
        matplotlib.pyplot.xcorr function

    Output
    ------
    Output will be in the format --> lag, corrarray

    lag: ndarray
        lag indices
    corrarray: ndarray
        cross correlation
    """

    #plotting
    corr = plt.xcorr(sig1, sig2, **kwargs)
    plt.title("Cross correlation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation coefficient")
    plt.show()

    corr = np.array(corr)
    lag = corr[0,]
    corrarray = corr[1,]

    return lag, corrarray


def acorr(signal, **kwargs):
    """
    Calculate the auto-correlation of a signal.

    Input parameters
    ----------------
    signal: ndarray
        signal whose auto-correlation is to be calculated
    **kwargs : dict, optional
        Additional keyword arguments are passed to the underlying
        matplotlib.pyplot.xcorr function

    Output
    ------
    Output will be in the format --> lag, acorrarray

    lag: ndarray
        lag indices
    acorrarray: ndarray
        auto correlation
    """

    #plotting
    corr = plt.xcorr(signal, signal, **kwargs)
    plt.title("Auto correlation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation coefficient")
    plt.show()

    corr = np.array(corr)
    lag = corr[0,]
    acorrarray = corr[1,]

    return lag, acorrarray


def psd(signal, fs=1000.0, plot='yes', **kwargs):
    """
    Estimate Power Spectral Density using periodogram.

    Input parameters
    ----------------
    signal: ndarray
        input signal to compute PSD
    fs: int or float, optional
        sampling rate; default set to 1000 Hz
    plot: str - yes or no, optional
        plot the PSD or not; default set to yes
    **kwargs : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal.periodogram function

    Output
    ------
    Output will be in the format --> freq, pxx

    freq: ndarray
        array of sample frequencies
    pxx: ndarray
        power spectral density of signal
    """

    freq, pxx = sp.signal.periodogram(signal, fs=fs, **kwargs)

    #plotting
    if plot in ['yes', 'Yes']:
        plt.figure(figsize=(10, 4))
        plt.semilogy(freq, pxx)
        plt.title("Power Spectral Density")
        plt.xlabel("Frequency")
        plt.ylabel("Power Spectral Density")
        plt.show()

    return freq, pxx


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

    #plotting
    if plot in ['yes', 'Yes']:
        plt.figure(figsize=(12, 4))
        time = np.arange(0, len(signal)/fs, 1/fs)
        plt.plot(time, signal, label="Raw signal")
        plt.title("Full wave rectified signal")
        plt.plot(time, rectifiedsig, c='#ff7f0e', label="Rectified signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    return rectifiedsig


def envelope(signal, fs=1000, order=2, cutoff=10, filter='butter', plot='Yes', **kwargs):
    """
    Linear envelope of the input signal, computed by low pass fitering the rectified signal.

    Input parameters
    ----------------
    signal: ndarray
        input filtered signal
    fs: int or float, optional
        sampling rate; default set to 1000 Hz
    order: int, optional
        the order of the filter; default set to 2
    cutoff: int or float, optional
        the critical frequency; default set to 10 Hz
    filter: str, optional
        type of IIR filter; default set to butter
        types: 'butter' (Butterworth), ‘cheby1’ (Chebyshev I), ‘cheby2’ (Chebyshev II),
        ‘ellip’ (Cauer/elliptic), ‘bessel’ (Bessel/Thomson)
    plot:  str - yes or no, optional
        plot the linear envelope or not; default set to yes
    **kwargs: dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal function

    Output
    ------
    Output will be in the format --> linenv

    linenv: ndarray
        linear envelope of the input signal
    """

    temp = rectify(signal, plot='No')
    linenv = iir(temp, fs=fs, ordern=order, cutoff=cutoff/2, ftype='lowpass',
                 filter=filter, plot='No', **kwargs)

    #plotting
    if plot in ['yes', 'Yes']:
        time = np.arange(0, len(signal)/fs, 1/fs)
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal, label="Input signal")
        plt.plot(time, linenv, label="Envelope", c='#ff7f0e', linewidth=3)
        plt.title("Envelope")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    return linenv


def rms(input):
    """
    This function returns the Root Mean Square of the input.

    Input parameters
    ----------------
    input: ndarray
        the input whose RMS is to be determined

    Output
    ------
    Output will be in the format --> rmsq

    rmsq: int or float
        the root mean square of the input
    """

    rmsq = np.sqrt(np.sum(input**2)/input.size)

    return rmsq


def rms_sig(signal, window=None, fs=1000, plot='yes'):
    """
    This function returns the Root Mean Square of the signal.

    Input parameters
    ----------------
    signal: ndarray
        the input signal whose RMS is to be determined
    window: int
        the size of the window for calculation RMS
        Example: Window size of 10 will have a moving window of 10 data points
    fs: int or float, optional
        sampling rate; default set to 1000 Hz
    plot: str - yes or no, optional
        plot the RMS of the signal or not; default set to yes

    Output
    ------
    Output will be in the format --> rms_signal

    rms_signal: ndarray
        the root mean square of the input signal
    """
    try:
        window = int(window)
    except TypeError:
        raise TypeError("rms_sig missing an argument: window")

    length = signal.size
    rms_signal = []
    for i in range(0, length-window, window):
        temp = rms(signal[i:(i+window)])
        rms_signal.append(temp)

    rms_signal = np.array(rms_signal)

    #plotting
    if plot in ['yes', 'Yes']:
        time = signal.size/fs
        t = np.arange(0, time, 1/fs)
        t1 = np.linspace(0, time, rms_signal.size)
        plt.figure(figsize=(12, 3))
        plt.plot(t, signal, label='Raw signal')
        plt.plot(t1, rms_signal, c='#ff7f0e', label='RMS signal', linewidth=3)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Root Mean Square")
        plt.legend()
        plt.show()

    return rms_signal


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
        signal = iir(signal, plot='No')

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
        emg = iir(signal=signal, fs=fs, plot='No')
        emg_rect = rectify(emg, fs=fs, plot='No')
        mvic = iir(signal=mvic, fs=fs, plot='No')
        mvic_rect = rectify(mvic, fs=fs, plot='No')

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
