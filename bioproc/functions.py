"""
Contains the functions for processing signals.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from biosppy.signals import tools

def sinewave(amp=1, freq=1, time=10, fs=1000, phi=0, offset=0, plot='no', tarr='no'):
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
        sampling frequency; set to 1000 Hz by default
    phi: int or float, optional
        phase (angle); set to 0 degrees by default
    offset: int or float, optional
        bias; set to 1 by default
    plot: str - yes/Y or no/N (non-case sensitive), optional
        plot the sine wave or not; default set to no
    tarr: str - yes/Y or no/N (non-case sensitive), optional
        return the time array or not; default set to no

    Output
    ------
    Output will be in the format --> t,sine

    t: int or float
        time of sine wave in secs
    sine: sine wave
        amplitude of the sine wave
    """

    #plot error
    if plot in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'NO', 'YES']:
        pass
    elif isinstance(plot, str):
        raise ValueError("plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #time error
    if tarr in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'NO', 'YES']:
        pass
    elif isinstance(tarr, str):
        raise ValueError("plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #error for other variables
    try:
        amp = float(amp)
        freq = float(freq)
        time = float(time)
        fs = float(fs)
        phi = float(phi)
        offset = float(offset)
    except ValueError:
        raise ValueError("amp, freq, time, fs, phi, offset must be int or float.")
    except TypeError:
        raise TypeError("amp, freq, time, fs, phi, offset must be int or float.")

    #function
    pi = 3.14
    t = np.arange(0, time, 1/fs)
    sine = offset + amp * (np.sin((2*pi*freq*t) + phi))

    #plotting
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
        plt.figure(figsize=(12, 4))
        plt.plot(t, sine)
        plt.title("Sine wave")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    #return time array or not
    if tarr in ['yes', 'Yes', 'Y', 'y', 'YES']:
        return t, sine

    return sine


def sinenoise(amp=1, freq=1, time=10, fs=1000, phi=0, offset=0,
              noise=1, plot='no', tarr='no', seed=1):
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
    plot: str - yes/Y or no/N (non-case sensitive), optional
        plot the sine wave or not; default set to no
    tarr: str - yes/Y or no/N (non-case sensitive), optional
        return the time array or not; default set to no
    seed: int, optional
        random generator seed; default set to 1

    Output
    ------
    Output will be in the format --> t,sine

    t: ndarray
       time array
    sine: sine wave
        amplitude of the sine wave
    """

    #plot error
    if plot in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'YES', 'NO']:
        pass
    elif isinstance(plot, str):
        raise ValueError("plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #time error
    if tarr in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'YES', 'NO']:
        pass
    elif isinstance(tarr, str):
        raise ValueError("plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #error for other variables
    try:
        amp = float(amp)
        freq = float(freq)
        time = float(time)
        fs = float(fs)
        phi = float(phi)
        offset = float(offset)
        noise = float(noise)
        seed = int(seed)
    except ValueError:
        raise ValueError("amp, freq, time, fs, phi, offset, noise, seed must be int or float.")
    except TypeError:
        raise TypeError("amp, freq, time, fs, phi, offset, noise, seed must be int or float.")

    #function
    pi = 3.14
    t = np.arange(0, time, 1/fs)
    np.random.seed(seed=seed)
    sine = offset + amp * (np.sin((2*pi*freq*t) + phi)) + (noise *(np.random.randn(len(t))))

    #plotting
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
        plt.figure(figsize=(12, 4))
        plt.plot(t, sine)
        plt.title("Sine wave with noise")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    #return time array or not
    if tarr in ['yes', 'Yes', 'Y', 'y', 'YES']:
        return t, sine

    return sine


def fft(signal, fs=1000, plot='yes', **kwargs):
    """
    This function is used to calculate the discrete fourier transform of the sine wave.

    Input parameters
    ----------------
    signal: ndarray
        the input signal whose DFT is to be determined
    fs: int or float, optional
        sampling frequency; default set to 1000 Hz
    plot: str - yes/Y or no/N (non-case sensitive), optional
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

    #sampling frequency error
    try:
        fs = float(fs)
    except TypeError:
        raise TypeError("sampling frequency (fs) must be int or float.")
    except ValueError:
        raise ValueError("sampling frequency (fs) must be int or float.")

    #plot error
    if plot in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'YES', 'NO']:
        pass
    elif isinstance(plot, str):
        raise ValueError("plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #signal error
    if isinstance(signal, (list, np.ndarray)):
        signal = np.array(signal)
    elif isinstance(signal, (str, complex)):
        raise TypeError("signal should be a list or numpy array.")
    else:
        raise ValueError("signal should be a list or numpy array.")

    #fft
    fourier = np.fft.fft(signal, **kwargs)
    N = signal.size
    amp = np.linspace(0, fs, N)

    #plotting
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
        plt.title("FFT")
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency (Hz)")
        plt.bar(amp[:N // 2], np.abs(fourier)[:N // 2]*1/N, width=1.5)
        plt.tight_layout()
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
        number of zeros to be added to the signal; default set to 0

    Output
    ------
    Output will be in the format --> padarray

    padarray: ndarray
        zero padded signal
    """

    #signal error
    if isinstance(signal, (list, np.ndarray)):
        signal = np.array(signal)
    elif isinstance(signal, int):
        raise ValueError("signal should be a list or numpy array.")
    else:
        raise TypeError("signal should be a list or numpy array.")

    for i in signal:
        if isinstance(i, complex):
            raise ValueError("signal cannot contain complex elements.")

    #size error
    if size <= 0:
        raise ValueError("size should be greater than zero.")

    try:
        size = int(abs(size))
    except TypeError:
        raise TypeError("size should be an int")

    #padding
    padarray = np.concatenate((signal, np.zeros(size)))

    return padarray


def padsize(signal, returnarr='Yes'):
    """
    This function determines the size of the input signal, suggests
    sizes for zero padding such that total size is a power of 2.

    Input parameters
    ----------------
    signal: ndarray
        the input signal
    returnarr: str - yes/Y or no/N (non-case sensitive), optional
        return zero padded array or not; default set to yes

    Output
    ------
    Output will be in the format --> padarray

    padarray: ndarray
        zero padded array; size will be a power of 2
    """

    #signal error
    if isinstance(signal, (list, np.ndarray)):
        signal = np.array(signal)
    elif isinstance(signal, (int, float)):
        raise ValueError("signal should be a list or numpy array.")
    else:
        raise TypeError("signal should be a list or numpy array.")

    for i in signal:
        if isinstance(i, complex):
            raise ValueError("signal cannot contain complex elements.")

    #return array error
    if returnarr in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'YES', 'NO']:
        pass
    elif isinstance(returnarr, str):
        raise ValueError("plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #determine size for padding and zero pad
    siz = signal.size
    binary = np.binary_repr(siz)
    if binary.count("1") == 1:
        temp = np.log(siz)//np.log(2)
        temp_2 = int(2**int(temp))
        if returnarr in ['No', 'no', 'N', 'n', 'NO']:
            print("""The size of the signal is {}, which is a power of 2 (2^{})={}.
            \nSuggestion:\nAdd {} more zeros using the padding function"""
                                    .format(siz, int(temp), temp_2, siz))
        elif returnarr in ['Yes', 'yes', 'Y', 'y', 'YES']:
            padarray = padding(signal, size=int(siz))

            return padarray

    else:
        temp = np.log(siz)//np.log(2)
        diff_1 = int((2**(temp+1)) - (2**(temp)))
        diff = int(2**(temp+1)) - siz
        temp_2 = int(2**(temp+1))

        if diff >= diff_1//2:
            if returnarr in ['No', 'no', 'N', 'n', 'NO']:
                print("""The size of the signal is {}. The closest power of 2 is 2^{}={}.
                \nSuggestion:\nAdd {} more zeros to bring it to the closest power of 2."""
                      .format(siz, int(temp+1), temp_2, diff))
            elif returnarr in ['Yes', 'yes', 'Y', 'y', 'YES']:
                padarray = padding(signal, size=int(diff))

                return padarray

        else:
            temp1 = diff+2**(temp+1)
            if returnarr in ['No', 'no', 'N', 'n', 'NO']:
                print("""The size of the signal is {}. The closest power of 2 is 2^{}={}.
                \nSuggestion:\nAdd {} more zeros. Bring it to {}. Better solution: add {} zeros."""
                      .format(siz, int(temp+1), int(2**(temp+1)), diff, int(2**(temp+1)), temp1))
            elif returnarr in ['Yes', 'yes', 'Y', 'y', 'YES']:
                padarray = padding(signal, size=int(temp1))

                return padarray


def window(kernel, size=0, **kwargs):
    """
    Return a window for the given parameters.

    Input parameters
    ----------------
    kernel : str
        Type of window to create.
    size : int
        Size of the window; default set to 0.
    **kwargs : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal.windows function

    Output
    ------
    Output will be in the format --> windows

    windows : ndarray
        Created window.
    """

    #kernel error
    if isinstance(kernel, str):
        pass
    else:
        raise TypeError("kernel must be a string (str).")

    #size error
    if size <= 0:
        raise ValueError("size should be greater than zero.")

    try:
        size = int(abs(size))
    except TypeError:
        raise TypeError("size should be an int")

    #get window
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
    plot:  str - yes/Y or no/N (non-case sensitive), optional
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

    #cutoff error
    try:
        cutoff = float(cutoff)
    except TypeError:
        cutoff = np.array(cutoff)
    except ValueError:
        raise ValueError("Cutoff can only be an int, float or numpy array")

    if isinstance(cutoff, np.ndarray):
        if cutoff.size == 2:
            if isinstance(cutoff[0], complex):
                raise TypeError("cutoff frequency cannot be complex")
            if isinstance(cutoff[-1], complex):
                raise TypeError("cutoff frequency cannot be complex")
        else:
            raise ValueError("cutoff must be a scalar (int or float) or 2 length sequence (list or numpy array)")

    #sampling frequency error
    try:
        fs = float(fs)
    except TypeError:
        raise TypeError("sampling frequency (fs) must be int or float")
    except ValueError:
        raise ValueError("sampling frequency (fs) must be int or float")

    #signal error
    if isinstance(signal, (list, np.ndarray)):
        signal = np.array(signal)
    elif isinstance(signal, (int, float)):
        raise ValueError("signal should be a list or numpy array")
    else:
        raise TypeError("signal should be a list or numpy array")

    for i in signal:
        if isinstance(i, complex):
            raise ValueError("signal cannot contain complex elements")

    #order error
    try:
        ordern = int(ordern)
    except TypeError:
        raise TypeError("order must be an int")
    except ValueError:
        raise ValueError("order must be an int")

    #filter type error
    if ftype in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        pass
    elif isinstance(ftype, str):
        raise ValueError("filter type must be 'lowpass', 'highpass', 'bandpass', 'bandstop'")
    else:
        raise TypeError("filter type must be a string")

    #IIR filter type error
    if filter in ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']:
        pass
    elif isinstance(filter, str):
        raise ValueError("IIR filter type must be 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'")
    else:
        raise TypeError("IIR filter type must be a string")

    #plot error
    if plot in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'YES', 'NO']:
        pass
    elif isinstance(plot, str):
        raise ValueError("plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #filtering
    filtersig = tools.filter_signal(signal, ftype=filter, band=ftype, order=ordern,
                                    frequency=cutoff, sampling_rate=fs, **kwargs)
    filtersig = filtersig['signal']

    #plotting
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
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
        plt.tight_layout()
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
    plot: str - yes/Y or no/N (non-case sensitive), optional
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

    #cutoff error
    try:
        cutoff = float(cutoff)
    except TypeError:
        cutoff = np.array(cutoff)
    except ValueError:
        raise ValueError("Cutoff can only be an int, float or numpy array")

    if isinstance(cutoff, np.ndarray):
        if cutoff.size == 2:
            if isinstance(cutoff[0], complex):
                raise TypeError("cutoff frequency cannot be complex")
            if isinstance(cutoff[-1], complex):
                raise TypeError("cutoff frequency cannot be complex")
        else:
            raise ValueError("cutoff must be a scalar (int or float) or 2 length sequence (list or numpy array)")

    #sampling frequency error
    try:
        fs = float(fs)
    except TypeError:
        raise TypeError("sampling frequency (fs) must be int or float")
    except ValueError:
        raise ValueError("sampling frequency (fs) must be int or float")

    #signal error
    if isinstance(signal, (list, np.ndarray)):
        signal = np.array(signal)
    elif isinstance(signal, (int, float)):
        raise ValueError("signal should be a list or numpy array")
    else:
        raise TypeError("signal should be a list or numpy array")

    for i in signal:
        if isinstance(i, complex):
            raise ValueError("signal cannot contain complex elements")

    #order error
    try:
        ordern = int(ordern)
    except TypeError:
        raise TypeError("order must be an int")
    except ValueError:
        raise ValueError("order must be an int")

    #filter type error
    if ftype in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        pass
    elif isinstance(ftype, str):
        raise ValueError("filter type must be 'lowpass', 'highpass', 'bandpass', 'bandstop'")
    else:
        raise TypeError("filter type must be a string")

    #plot error
    if plot in ['Yes', 'yes', 'No', 'no', 'Y', 'y', 'N', 'n', 'YES', 'NO']:
        pass
    elif isinstance(plot, str):
        raise ValueError("plot can be Yes/Y or No/N (non-case sensitive).")
    else:
        raise TypeError("plot must be a string - Yes/Y or No/N (non-case sensitive).")

    #filtering
    filtersig = tools.filter_signal(signal, ftype='FIR', band=ftype, order=ordern,
                                    frequency=cutoff, sampling_rate=fs, **kwargs)
    filtersig = filtersig['signal']

    #plotting
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
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
        plt.tight_layout()
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

    if isinstance(signal, (list, np.ndarray)):
        signal = np.array(signal)
    else:
        raise TypeError("signal should be a list or numpy array.")

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
    plot: str - yes/Y or no/N (non-case sensitive), optional
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
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
        plt.plot(time, signal, label="Signal")
        plt.plot(time, regression, c='#ff7f0e', label="Polynomial")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Polynomial fitting")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return regression


def xcorr(sig1, sig2, **kwargs):
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
    plt.tight_layout()
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
        matplotlib.pyplot.acorr function

    Output
    ------
    Output will be in the format --> lag, acorrarray

    lag: ndarray
        lag indices
    acorrarray: ndarray
        auto correlation
    """

    #plotting
    corr = plt.acorr(signal, **kwargs)
    plt.title("Auto correlation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation coefficient")
    plt.tight_layout()
    plt.show()

    corr = np.array(corr)
    lag = corr[0,]
    acorrarray = corr[1,]

    return lag, acorrarray


def correlogram(signal, **kwargs):
    """
    Returns the correlogram of the signal.

    Input parameters
    ----------------
    signal: ndarray
        signal whose correlogram is to be returned
    **kwargs : dict, optional
        Additional keyword arguments are passed to the underlying
        pandas.plotting.autocorrelation_plot function

    Output
    ------
    Output will be the correlogram.
    """

    pd.plotting.autocorrelation_plot(signal)

    return None


def psd(signal, fs=1000.0, plot='yes', **kwargs):
    """
    Estimate Power Spectral Density using periodogram.

    Input parameters
    ----------------
    signal: ndarray
        input signal to compute PSD
    fs: int or float, optional
        sampling rate; default set to 1000 Hz
    plot: str - yes/Y or no/N (non-case sensitive), optional
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
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
        plt.figure(figsize=(10, 4))
        plt.semilogy(freq, pxx)
        plt.title("Power Spectral Density")
        plt.xlabel("Frequency")
        plt.ylabel("Power Spectral Density")
        plt.tight_layout()
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
    plt: str - yes/Y or no/N (non-case sensitive), optional
        plot the rectified signal or not; default set to yes

    Output
    ------
    Output will be in the format --> rectifiedsig

    rectifiedsig: ndarray
        rectified signal array
    """

    rectifiedsig = np.abs(signal)

    #plotting
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
        plt.figure(figsize=(12, 4))
        time = np.arange(0, len(signal)/fs, 1/fs)
        plt.plot(time, signal, label="Raw signal")
        plt.title("Full wave rectified signal")
        plt.plot(time, rectifiedsig, c='#ff7f0e', label="Rectified signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
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
    plot:  str - yes/Y or no/N (non-case sensitive), optional
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
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
        time = np.arange(0, len(signal)/fs, 1/fs)
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal, label="Input signal")
        plt.plot(time, linenv, label="Envelope", c='#ff7f0e', linewidth=3)
        plt.title("Envelope")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
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


def rms_sig(signal, window_size, fs=1000, plot='yes'):
    """
    This function returns the Root Mean Square of the signal.

    Input parameters
    ----------------
    signal: ndarray
        the input signal whose RMS is to be determined
    window_size: int
        the size of the window for calculation RMS
        Example: Window size of 10 will have a moving window of 10 data points
    fs: int or float, optional
        sampling rate; default set to 1000 Hz
    plot: str - yes/Y or no/N (non-case sensitive), optional
        plot the RMS of the signal or not; default set to yes

    Output
    ------
    Output will be in the format --> rms_signal

    rms_signal: ndarray
        the root mean square of the input signal
    """
    try:
        window_size = int(window_size)
    except TypeError:
        raise TypeError("rms_sig missing an argument: wind")

    length = signal.size
    rms_signal = []
    for i in range(0, length-window_size, window_size):
        temp = rms(signal[i:(i+window_size)])
        rms_signal.append(temp)

    rms_signal = np.array(rms_signal)

    #plotting
    if plot in ['yes', 'Yes', 'Y', 'y', 'YES']:
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
        plt.tight_layout()
        plt.show()

    return rms_signal
