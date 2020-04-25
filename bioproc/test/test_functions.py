"""
test_functions.py

Author: Praveen Prabhakar KR
Email: praveenp@msu.edu

Module contains the tests for functions module.
"""

import pytest
import numpy as np
from .. import functions as fn


def test_sinewave():
    """
    Test for sinewave function in the functions module.
    """

    time = np.arange(0, 10, 1/1000)
    sinearr = np.sin(2*3.14*time)
    func = fn.sinewave()
    assert np.array_equal(func, sinearr)
    with pytest.raises(ValueError):
        fn.sinewave(amp='s')
    with pytest.raises(ValueError):
        fn.sinewave(freq='s')
    with pytest.raises(ValueError):
        fn.sinewave(time='s')
    with pytest.raises(ValueError):
        fn.sinewave(fs='s')
    with pytest.raises(TypeError):
        fn.sinewave(amp=[12, 12])
    with pytest.raises(TypeError):
        fn.sinewave(freq=[12, 12])
    with pytest.raises(TypeError):
        fn.sinewave(time=[12, 12])
    with pytest.raises(TypeError):
        fn.sinewave(fs=[12, 12])
    with pytest.raises(TypeError):
        fn.sinewave(offset=[12, 12])
    with pytest.raises(TypeError):
        fn.sinewave(phi=[12, 12])
    with pytest.raises(ValueError):
        fn.sinewave(plot='sa')
    with pytest.raises(TypeError):
        fn.sinewave(plot=123)
    with pytest.raises(ValueError):
        fn.sinewave(tarr='sa')
    with pytest.raises(TypeError):
        fn.sinewave(tarr=123)
    with pytest.raises(TypeError):
        fn.sinewave(amp=1+3j)
    with pytest.raises(TypeError):
        fn.sinewave(freq=1+3j)
    with pytest.raises(TypeError):
        fn.sinewave(time=1+3j)
    with pytest.raises(TypeError):
        fn.sinewave(fs=1+3j)
    with pytest.raises(TypeError):
        fn.sinewave(phi=1+3j)
    with pytest.raises(TypeError):
        fn.sinewave(offset=1+3j)
    with pytest.raises(TypeError):
        fn.sinewave(tarr=1+3j)
    with pytest.raises(TypeError):
        fn.sinewave(plot=1+3j)


def test_sinenoise():
    """
    Test for sinenoise function in the functions module.
    """

    time = np.arange(0, 10, 1/1000)
    np.random.seed(1)
    sinearr = np.sin(2*3.14*time) + (np.random.randn(len(time)))
    func = fn.sinenoise()

    assert np.array_equal(func, sinearr)
    with pytest.raises(ValueError):
        fn.sinenoise(amp='s')
    with pytest.raises(ValueError):
        fn.sinenoise(freq='s')
    with pytest.raises(ValueError):
        fn.sinenoise(time='s')
    with pytest.raises(ValueError):
        fn.sinenoise(fs='s')
    with pytest.raises(TypeError):
        fn.sinenoise(amp=[12, 12])
    with pytest.raises(TypeError):
        fn.sinenoise(freq=[12, 12])
    with pytest.raises(TypeError):
        fn.sinenoise(time=[12, 12])
    with pytest.raises(TypeError):
        fn.sinenoise(fs=[12, 12])
    with pytest.raises(TypeError):
        fn.sinenoise(seed=[12, 12])
    with pytest.raises(ValueError):
        fn.sinenoise(plot='sa')
    with pytest.raises(TypeError):
        fn.sinenoise(plot=123)
    with pytest.raises(ValueError):
        fn.sinenoise(tarr='sa')
    with pytest.raises(TypeError):
        fn.sinenoise(tarr=123)
    with pytest.raises(TypeError):
        fn.sinenoise(amp=1+3j)
    with pytest.raises(TypeError):
        fn.sinenoise(freq=1+3j)
    with pytest.raises(TypeError):
        fn.sinenoise(time=1+3j)
    with pytest.raises(TypeError):
        fn.sinenoise(fs=1+3j)
    with pytest.raises(TypeError):
        fn.sinenoise(phi=1+3j)
    with pytest.raises(TypeError):
        fn.sinenoise(offset=1+3j)
    with pytest.raises(TypeError):
        fn.sinenoise(noise=1+3j)
    with pytest.raises(TypeError):
        fn.sinenoise(tarr=1+3j)
    with pytest.raises(TypeError):
        fn.sinenoise(plot=1+3j)


def test_fft():
    """
    Test for fft function in the functions module.
    """

    with pytest.raises(TypeError):
        fn.fft(fn.sinewave(), fs=[21, 3])
    with pytest.raises(ValueError):
        fn.fft(fn.sinewave(), fs='sa')
    with pytest.raises(TypeError):
        fn.fft(fn.sinewave(), fs=1+3j)
    with pytest.raises(ValueError):
        fn.fft(signal=1)
    with pytest.raises(TypeError):
        fn.fft(signal='sa')
    with pytest.raises(TypeError):
        fn.fft(signal=1+3j)
    with pytest.raises(ValueError):
        fn.fft(signal=fn.sinewave(), plot='sa')
    with pytest.raises(TypeError):
        fn.fft(signal=fn.sinewave(), plot=123)


def test_padding():
    """
    Test for padding function in the functions module.
    """

    with pytest.raises(ValueError):
        fn.padding(signal=1, size=2)
    with pytest.raises(TypeError):
        fn.padding(signal='sa', size=2)
    with pytest.raises(TypeError):
        fn.padding(signal=1+3j, size=2)
    with pytest.raises(ValueError):
        fn.padding(signal=[1+3j], size=2)
    with pytest.raises(TypeError):
        fn.padding([1, 2, 3, 4, 5], size='sa')
    with pytest.raises(TypeError):
        fn.padding([1, 2, 3, 4, 5], size=[2, 4])
    with pytest.raises(ValueError):
        fn.padding([1, 2, 3, 4, 5], size=-1)
    with pytest.raises(TypeError):
        fn.padding([1, 2, 3, 4], size=1+3j)
    abab = [1, 2, 3, 4]
    baba = np.array([1, 2, 3, 4, 0, 0, 0, 0])
    assert np.array_equal(fn.padding(abab, 4), baba)


def test_padsize():
    """
    Test for padsize function in the functions module.
    """

    with pytest.raises(ValueError):
        fn.padsize(signal=1)
    with pytest.raises(ValueError):
        fn.padsize(signal=-1)
    with pytest.raises(TypeError):
        fn.padsize(signal='sa')
    with pytest.raises(TypeError):
        fn.padsize(signal=1+3j)
    with pytest.raises(ValueError):
        fn.padsize(signal=[1+3j])
    abab = [1, 2, 3, 4]
    baba = np.array([1, 2, 3, 4, 0, 0, 0, 0])
    assert np.array_equal(fn.padsize(abab), baba)
    assert fn.padsize(abab, 'No') is None
    abab = np.zeros(512)
    baba = fn.padsize(abab)
    assert baba.size == 1024
    abab = np.zeros(768)
    baba = fn.padsize(abab)
    assert baba.size == 1024
    abab = np.zeros(769)
    baba = fn.padsize(abab)
    assert baba.size == 2048
    with pytest.raises(ValueError):
        fn.padsize(abab, 'asda')
    with pytest.raises(TypeError):
        fn.padsize(abab, [1, 2])
    with pytest.raises(TypeError):
        fn.padsize(abab, 1+2j)


def test_window():
    """
    Test for window function in the functions module.
    """

    with pytest.raises(TypeError):
        fn.window(kernel=12, size=2)
    with pytest.raises(TypeError):
        fn.window(kernel=[1, 2], size=2)
    with pytest.raises(ValueError):
        fn.window(kernel='dummy', size=2)
    with pytest.raises(ValueError):
        fn.window(kernel='hamming')
    with pytest.raises(TypeError):
        fn.window(kernel='hamming', size='as')
    with pytest.raises(TypeError):
        fn.window(kernel='hamming', size=1+3j)
    with pytest.raises(TypeError):
        fn.window(kernel='hamming', size=[1, 2])


def test_iir():
    """
    Test for iir function in the functions module.
    """

    with pytest.raises(ValueError):
        fn.iir(signal=1)
    with pytest.raises(TypeError):
        fn.iir(signal='sa')
    with pytest.raises(TypeError):
        fn.iir(signal=1+3j)
    with pytest.raises(ValueError):
        fn.iir(signal=[1, 1+3j])
    signal = fn.sinewave()
    with pytest.raises(ValueError):
        fn.iir(signal, fs='s')
    with pytest.raises(TypeError):
        fn.iir(signal, fs=[1, 2])
    with pytest.raises(TypeError):
        fn.iir(signal, fs=1+3j)
    with pytest.raises(ValueError):
        fn.iir(signal, order=-1)
    with pytest.raises(ValueError):
        fn.iir(signal, order='s')
    with pytest.raises(TypeError):
        fn.iir(signal, order=1+3j)
    with pytest.raises(ValueError):
        fn.iir(signal, cutoff=[1, 2, 3])
    with pytest.raises(TypeError):
        fn.iir(signal, cutoff=[1+3j, 4])
    with pytest.raises(ValueError):
        fn.iir(signal, cutoff=-1)
    with pytest.raises(ValueError):
        fn.iir(signal, cutoff='s')
    with pytest.raises(ValueError):
        fn.iir(signal, cutoff=1+3j)
    with pytest.raises(ValueError):
        fn.iir(signal, plot='happy')
    with pytest.raises(TypeError):
        fn.iir(signal, plot=1223)
    with pytest.raises(ValueError):
        fn.iir(signal, ftype='yes')
    with pytest.raises(TypeError):
        fn.iir(signal, ftype=12)
    with pytest.raises(ValueError):
        fn.iir(signal, ftype='12')
    with pytest.raises(TypeError):
        fn.iir(signal, filter=12)
    with pytest.raises(ValueError):
        fn.iir(signal, filter='yes')
    with pytest.raises(ValueError):
        fn.iir(signal, filter='sa')
    with pytest.raises(TypeError):
        fn.iir(signal, dummy=1)


def test_fir():
    """
    Test for fir function in the functions module.
    """

    with pytest.raises(ValueError):
        fn.fir(signal=1)
    with pytest.raises(TypeError):
        fn.fir(signal='sa')
    with pytest.raises(TypeError):
        fn.fir(signal=1+3j)
    with pytest.raises(ValueError):
        fn.fir(signal=[1, 1+3j])
    signal = fn.sinewave()
    with pytest.raises(ValueError):
        fn.fir(signal, fs='s')
    with pytest.raises(TypeError):
        fn.fir(signal, fs=[1, 2])
    with pytest.raises(TypeError):
        fn.fir(signal, fs=1+3j)
    with pytest.raises(ValueError):
        fn.fir(signal, order=-1)
    with pytest.raises(ValueError):
        fn.fir(signal, order='s')
    with pytest.raises(ValueError):
        fn.fir(signal, cutoff=[1, 2, 3])
    with pytest.raises(TypeError):
        fn.fir(signal, cutoff=[1+3j, 4])
    with pytest.raises(ValueError):
        fn.fir(signal, cutoff=-1)
    with pytest.raises(ValueError):
        fn.fir(signal, cutoff='s')
    with pytest.raises(ValueError):
        fn.fir(signal, cutoff=1+3j)
    with pytest.raises(ValueError):
        fn.fir(signal, plot='happy')
    with pytest.raises(TypeError):
        fn.fir(signal, plot=1223)
    with pytest.raises(ValueError):
        fn.fir(signal, ftype='yes')
    with pytest.raises(TypeError):
        fn.fir(signal, ftype=12)
    with pytest.raises(ValueError):
        fn.fir(signal, ftype='12')
    with pytest.raises(TypeError):
        fn.fir(signal, dummy=1)


def test_movavg():
    """
    Test for movavg function in the functions module.
    """

    signal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    check = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    mov = fn.movavg(signal)
    assert np.array_equal(mov, check)
    check = np.array([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    mov = fn.movavg(signal, 4)
    assert np.array_equal(mov, check)
    mov = fn.movavg(signal, 4.9)
    assert np.array_equal(mov, check)
    with pytest.raises(ValueError):
        fn.movavg(signal=1)
    with pytest.raises(TypeError):
        fn.movavg(signal='sa')
    with pytest.raises(TypeError):
        fn.movavg(signal=1+3j)
    with pytest.raises(ValueError):
        fn.movavg(signal=[1, 1+3j])
    signal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        fn.movavg(signal, -1)
    with pytest.raises(ValueError):
        fn.movavg(signal, 'a')
    with pytest.raises(TypeError):
        fn.movavg(signal, 1+3j)


def test_polyfit():
    """
    Test for polyfit function in the functions module.
    """

    time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    signal = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    para = np.polyfit(time, signal, 2)
    func = np.poly1d(para)
    reg = func(time)
    reg1 = fn.polyfit(time, signal, 2, plot='n')
    assert np.array_equal(reg, reg1)
    with pytest.raises(ValueError):
        fn.polyfit(time, signal, degree=2, plot='happy')
    with pytest.raises(TypeError):
        fn.polyfit(time, signal, degree=2, plot=1223)
    with pytest.raises(ValueError):
        fn.polyfit(time, signal, degree=-2)
    with pytest.raises(ValueError):
        fn.polyfit(time=time, signal=1, degree=2)
    with pytest.raises(TypeError):
        fn.polyfit(time=time, signal='sa', degree=2)
    with pytest.raises(TypeError):
        fn.polyfit(time=time, signal=1+3j, degree=2)
    with pytest.raises(ValueError):
        fn.polyfit(time=time, signal=[1, 1+3j], degree=2)
    signal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        fn.polyfit(time=1, signal=signal, degree=2)
    with pytest.raises(TypeError):
        fn.polyfit(time='sa', signal=signal, degree=2)
    with pytest.raises(TypeError):
        fn.polyfit(time=1+3j, signal=signal, degree=2)
    with pytest.raises(ValueError):
        fn.polyfit(time=[1, 1+3j], signal=signal, degree=2)


def test_xcorr():
    """
    Test for xcorr function in the functions module.
    """

    sig1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sig2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        fn.xcorr(sig1, sig2=1)
    with pytest.raises(TypeError):
        fn.xcorr(sig1, sig2='sa')
    with pytest.raises(TypeError):
        fn.xcorr(sig1, sig2=1+3j)
    with pytest.raises(ValueError):
        fn.xcorr(sig1, sig2=[1, 1+3j])
    with pytest.raises(ValueError):
        fn.xcorr(sig1=1, sig2=sig2)
    with pytest.raises(TypeError):
        fn.xcorr(sig1='sa', sig2=sig2)
    with pytest.raises(TypeError):
        fn.xcorr(sig1=1+3j, sig2=sig2)
    with pytest.raises(ValueError):
        fn.xcorr(sig1=[1, 1+3j], sig2=sig2)


def test_acorr():
    """
    Test for acorr function in the functions module.
    """

    with pytest.raises(ValueError):
        fn.acorr(signal=1)
    with pytest.raises(TypeError):
        fn.acorr(signal='sa')
    with pytest.raises(TypeError):
        fn.acorr(signal=1+3j)
    with pytest.raises(ValueError):
        fn.acorr(signal=[1, 1+3j])


def test_correlogram():
    """
    Test for correlogram function in the functions module.
    """

    with pytest.raises(ValueError):
        fn.correlogram(signal=1)
    with pytest.raises(TypeError):
        fn.correlogram(signal='sa')
    with pytest.raises(TypeError):
        fn.correlogram(signal=1+3j)
    with pytest.raises(ValueError):
        fn.correlogram(signal=[1, 1+3j])


def test_psd():
    """
    Test for psd function in the functions module.
    """

    with pytest.raises(ValueError):
        fn.psd(signal=1)
    with pytest.raises(TypeError):
        fn.psd(signal='sa')
    with pytest.raises(TypeError):
        fn.psd(signal=1+3j)
    with pytest.raises(ValueError):
        fn.psd(signal=[1, 1+3j])
    signal = fn.sinewave()
    with pytest.raises(ValueError):
        fn.psd(signal, fs='s')
    with pytest.raises(TypeError):
        fn.psd(signal, fs=[1, 2])
    with pytest.raises(TypeError):
        fn.psd(signal, fs=1+3j)
    with pytest.raises(ValueError):
        fn.iir(signal, plot='happy')
    with pytest.raises(TypeError):
        fn.iir(signal, plot=1223)


def test_rectify():
    """
    Test for rectify function in the functions module.
    """

    signal = fn.sinewave()
    with pytest.raises(ValueError):
        fn.rectify(signal, plot='happy')
    with pytest.raises(TypeError):
        fn.rectify(signal, plot=1223)
    with pytest.raises(ValueError):
        fn.rectify(signal, fs='s')
    with pytest.raises(TypeError):
        fn.rectify(signal, fs=[1, 2])
    with pytest.raises(TypeError):
        fn.rectify(signal, fs=1+3j)
    with pytest.raises(ValueError):
        fn.rectify(signal=1)
    with pytest.raises(TypeError):
        fn.rectify(signal='sa')
    with pytest.raises(TypeError):
        fn.rectify(signal=1+3j)
    with pytest.raises(ValueError):
        fn.rectify(signal=[1, 1+3j])
    sample = np.array([-1, 1, -2, -5, -6, 4])
    answer = np.array([1, 1, 2, 5, 6, 4])
    ans = fn.rectify(sample, plot='N')
    assert np.array_equal(ans, answer)
    sample = np.array([-1, -1, -2, -5, -6, -4])
    answer = np.array([1, 1, 2, 5, 6, 4])
    ans = fn.rectify(sample, plot='N')
    assert np.array_equal(ans, answer)
    sample = np.array([1, 1, 2, 5, 6, 4])
    answer = np.array([1, 1, 2, 5, 6, 4])
    ans = fn.rectify(sample, plot='N')
    assert np.array_equal(ans, answer)


def test_envelope():
    """
    Test for envelope function in the functions module.
    """

    with pytest.raises(ValueError):
        fn.envelope(signal=1)
    with pytest.raises(TypeError):
        fn.envelope(signal='sa')
    with pytest.raises(TypeError):
        fn.envelope(signal=1+3j)
    with pytest.raises(ValueError):
        fn.envelope(signal=[1, 1+3j])
    signal = fn.sinewave()
    with pytest.raises(ValueError):
        fn.envelope(signal, fs='s')
    with pytest.raises(TypeError):
        fn.envelope(signal, fs=[1, 2])
    with pytest.raises(TypeError):
        fn.envelope(signal, fs=1+3j)
    with pytest.raises(ValueError):
        fn.envelope(signal, order=-1)
    with pytest.raises(ValueError):
        fn.envelope(signal, order='s')
    with pytest.raises(ValueError):
        fn.envelope(signal, cutoff=[1, 2, 3])
    with pytest.raises(TypeError):
        fn.envelope(signal, cutoff=[1+3j, 4])
    with pytest.raises(ValueError):
        fn.envelope(signal, cutoff=-1)
    with pytest.raises(ValueError):
        fn.envelope(signal, cutoff='s')
    with pytest.raises(ValueError):
        fn.envelope(signal, cutoff=1+3j)
    with pytest.raises(ValueError):
        fn.envelope(signal, plot='happy')
    with pytest.raises(TypeError):
        fn.envelope(signal, plot=1223)
    with pytest.raises(ValueError):
        fn.envelope(signal, filter='yes')
    with pytest.raises(TypeError):
        fn.envelope(signal, filter=12)
    with pytest.raises(ValueError):
        fn.envelope(signal, filter='12')
    with pytest.raises(TypeError):
        fn.envelope(signal, dummy=1)



def test_rms():
    """
    Test for rms function in the functions module.
    """

    sample = [0, 0, 6, 8]
    assert fn.rms(sample) == 5
    with pytest.raises(ValueError):
        fn.rms(signal=1)
    with pytest.raises(TypeError):
        fn.rms(signal='sa')
    with pytest.raises(TypeError):
        fn.rms(signal=1+3j)
    with pytest.raises(ValueError):
        fn.rms(signal=[1, 1+3j])


def test_rms_sig():
    """
    Test for rms_sig function in the functions module.
    """

    sample = [0, 0, 6, 8, 0, 0, 6, 8, 0, 0, 6, 8]
    ans = np.array([5., 5., 5., 5., 5., 5., 5., 5., 5.])
    assert np.array_equal(fn.rms_sig(sample, window_size=4, plot='N'), ans)
    with pytest.raises(ValueError):
        fn.rms_sig(signal=1, window_size=4, plot='N')
    with pytest.raises(TypeError):
        fn.rms_sig(signal='sa', window_size=4, plot='N')
    with pytest.raises(TypeError):
        fn.rms_sig(signal=1+3j, window_size=4, plot='N')
    with pytest.raises(ValueError):
        fn.rms_sig(signal=[1, 1+3j], window_size=4, plot='N')
    sample = [0, 0, 6, 8, 0, 0, 6, 8, 0, 0, 6, 8]
    with pytest.raises(ValueError):
        fn.rms_sig(signal=sample, window_size=1, plot='N')
    with pytest.raises(ValueError):
        fn.rms_sig(signal=sample, window_size='sa', plot='N')
    with pytest.raises(TypeError):
        fn.rms_sig(signal=sample, window_size=1+3j, plot='N')
    with pytest.raises(TypeError):
        fn.rms_sig(signal=sample, window_size=[1+3j], plot='N')
    signal = fn.sinewave()
    with pytest.raises(ValueError):
        fn.rms_sig(signal, window_size=4, fs='s', plot='N')
    with pytest.raises(TypeError):
        fn.rms_sig(signal, window_size=4, fs=[1, 2], plot='N')
    with pytest.raises(TypeError):
        fn.rms_sig(signal, window_size=4, fs=1+3j, plot='N')
    with pytest.raises(ValueError):
        fn.rms_sig(signal, window_size=4, plot='happy')
    with pytest.raises(TypeError):
        fn.rms_sig(signal, window_size=4, plot=1223)
