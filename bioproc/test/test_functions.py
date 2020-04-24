import pytest
import numpy as np
from .. import functions as fn


def test_sinewave():
    t = np.arange(0,10,1/1000)
    sinearr = np.sin(2*3.14*t)
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
    with pytest.raises(ValueError):
        fn.sinewave(offset='s')  
    with pytest.raises(ValueError):
        fn.sinewave(phi='s')
    with pytest.raises(NameError):
        fn.sinewave(amp=s)
    with pytest.raises(NameError):
        fn.sinewave(freq=s)  
    with pytest.raises(NameError):
        fn.sinewave(time=s)  
    with pytest.raises(NameError):
        fn.sinewave(fs=s)
    with pytest.raises(NameError):
        fn.sinewave(offset=s)  
    with pytest.raises(NameError):
        fn.sinewave(phi=s)  
    with pytest.raises(TypeError):
        fn.sinewave(amp=[12,12])
    with pytest.raises(TypeError):
        fn.sinewave(freq=[12,12])  
    with pytest.raises(TypeError):
        fn.sinewave(time=[12,12])  
    with pytest.raises(TypeError):
        fn.sinewave(fs=[12,12])
    with pytest.raises(TypeError):
        fn.sinewave(offset=[12,12])  
    with pytest.raises(TypeError):
        fn.sinewave(phi=[12,12])
    with pytest.raises(ValueError):
        fn.sinewave(plot='sa')
    with pytest.raises(TypeError):
        fn.sinewave(plot=123)
    with pytest.raises(NameError):
        fn.sinewave(plot=s)
    with pytest.raises(ValueError):
        fn.sinewave(tarr='sa')
    with pytest.raises(TypeError):
        fn.sinewave(tarr=123)
    with pytest.raises(NameError):
        fn.sinewave(tarr=s)
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
    t = np.arange(0,10,1/1000)
    np.random.seed(1)
    sinearr = np.sin(2*3.14*t) + (np.random.randn(len(t)))
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
    with pytest.raises(ValueError):
        fn.sinenoise(offset='s')  
    with pytest.raises(ValueError):
        fn.sinenoise(phi='s')
    with pytest.raises(ValueError):
        fn.sinenoise(seed='s')
    with pytest.raises(NameError):
        fn.sinenoise(amp=s)
    with pytest.raises(NameError):
        fn.sinenoise(freq=s)  
    with pytest.raises(NameError):
        fn.sinenoise(time=s)  
    with pytest.raises(NameError):
        fn.sinenoise(fs=s)
    with pytest.raises(NameError):
        fn.sinenoise(offset=s)  
    with pytest.raises(NameError):
        fn.sinenoise(phi=s)
    with pytest.raises(NameError):
        fn.sinenoise(seed=s)
    with pytest.raises(TypeError):
        fn.sinenoise(amp=[12, 12])
    with pytest.raises(TypeError):
        fn.sinenoise(freq=[12, 12])  
    with pytest.raises(TypeError):
        fn.sinenoise(time=[12, 12])  
    with pytest.raises(TypeError):
        fn.sinenoise(fs=[12, 12])
    with pytest.raises(TypeError):
        fn.sinenoise(offset=[12, 12])  
    with pytest.raises(TypeError):
        fn.sinenoise(phi=[12, 12])
    with pytest.raises(TypeError):
        fn.sinenoise(seed=[12, 12])
    with pytest.raises(ValueError):
        fn.sinenoise(plot='sa')
    with pytest.raises(TypeError):
        fn.sinenoise(plot=123)
    with pytest.raises(NameError):
        fn.sinenoise(plot=s)
    with pytest.raises(ValueError):
        fn.sinenoise(tarr='sa')
    with pytest.raises(TypeError):
        fn.sinenoise(tarr=123)
    with pytest.raises(NameError):
        fn.sinenoise(tarr=s)
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
        fn.sinewave(noise=1+3j)
    with pytest.raises(TypeError):
        fn.sinewave(tarr=1+3j)
    with pytest.raises(TypeError):
        fn.sinewave(plot=1+3j)


def test_fft():
    with pytest.raises(TypeError):
        fn.fft(fn.sinewave(), fs=[21, 3])
    with pytest.raises(ValueError):
        fn.fft(fn.sinewave(), fs='sa')
    with pytest.raises(TypeError):
        fn.fft(fn.sinewave(), fs=1+3j)
    with pytest.raises(NameError):
        fn.fft(fn.sinewave(), fs=sa)
    with pytest.raises(ValueError):
        fn.fft(signal=1)
    with pytest.raises(TypeError):
        fn.fft(signal='sa')
    with pytest.raises(TypeError):
        fn.fft(signal=1+3j)
    with pytest.raises(NameError):
        fn.fft(signal=sa)
    with pytest.raises(ValueError):
        fn.sinenoise(plot='sa')
    with pytest.raises(TypeError):
        fn.sinenoise(plot=123)
    with pytest.raises(NameError):
        fn.sinenoise(plot=s)


def test_padding():
    with pytest.raises(ValueError):
        fn.padding(signal=1, size=2)
    with pytest.raises(TypeError):
        fn.padding(signal='sa', size=2)
    with pytest.raises(NameError):
        fn.padding(signal=sa, size=2)
    with pytest.raises(TypeError):
        fn.padding(signal=1+3j, size=2)
    with pytest.raises(ValueError):
        fn.padding(signal=[1+3j], size=2)
    with pytest.raises(NameError):
        fn.padding([1, 2, 3, 4, 5, 6], size=sa)
    with pytest.raises(TypeError):
        fn.padding([1, 2, 3, 4, 5], size='sa')
    with pytest.raises(TypeError):
        fn.padding([1, 2, 3, 4, 5], size=[2,4])
    with pytest.raises(ValueError):
        fn.padding([1, 2, 3, 4, 5], size=-1)
    with pytest.raises(TypeError):
        fn.padding([1, 2, 3, 4], size=1+3j)
    a = [1, 2, 3, 4]
    b = np.array([1, 2, 3, 4, 0, 0, 0, 0])
    assert np.array_equal(fn.padding(a, 4), b)


def test_padsize():
    with pytest.raises(ValueError):
        fn.padsize(signal=1)
    with pytest.raises(ValueError):
        fn.padsize(signal=-1)
    with pytest.raises(TypeError):
        fn.padsize(signal='sa')
    with pytest.raises(NameError):
        fn.padsize(signal=sa)
    with pytest.raises(TypeError):
        fn.padsize(signal=1+3j)
    with pytest.raises(ValueError):
        fn.padsize(signal=[1+3j])
    a = [1,2,3,4]
    b = np.array([1, 2, 3, 4, 0, 0, 0, 0])
    assert np.array_equal(fn.padsize(a), b)
    assert fn.padsize(a, 'No') == None
    a = np.zeros(512)
    b = fn.padsize(a)
    assert b.size == 1024
    a = np.zeros(768)
    b = fn.padsize(a)
    assert b.size == 1024
    a = np.zeros(769)
    b = fn.padsize(a)
    assert b.size == 2048


def test_window():
    with pytest.raises(TypeError):
        fn.window(kernel=12, size=2)
    with pytest.raises(TypeError):
        fn.window(kernel=[1,2], size=2)
    with pytest.raises(ValueError):
        fn.window(kernel='dummy', size=2)
    with pytest.raises(NameError):
        fn.window(kernel=dummy, size=2)
    with pytest.raises(ValueError):
        fn.window(kernel='hamming')
    with pytest.raises(TypeError):
        fn.window(kernel='hamming', size='as')
    with pytest.raises(TypeError):
        fn.window(kernel='hamming', size=1+3j)
    with pytest.raises(TypeError):
        fn.window(kernel='hamming', size=[1, 2])


def test_iir():
    with pytest.raises(ValueError):
        fn.iir(signal=1)
    with pytest.raises(TypeError):
        fn.iir(signal='sa')
    with pytest.raises(NameError):
        fn.iir(signal=sa)
    with pytest.raises(TypeError):
        fn.iir(signal=1+3j)
    with pytest.raises(ValueError):
        fn.iir(signal=[1, 1+3j])
    signal = fn.sinewave()
    with pytest.raises(ValueError):
        fn.iir(signal, fs='s')
    with pytest.raises(NameError):
        fn.iir(signal, fs=s)
    with pytest.raises(TypeError):
        fn.iir(signal, fs=[1, 2])
    with pytest.raises(TypeError):
        fn.iir(signal, fs=1+3j)
    with pytest.raises(ValueError):
        fn.iir(signal, ordern=-1)
    with pytest.raises(NameError):
        fn.iir(signal, ordern=s)
    with pytest.raises(ValueError):
        fn.iir(signal, cutoff=[1, 2, 3])
    with pytest.raises(TypeError):
        fn.iir(signal, cutoff=[1+3j, 4])
    with pytest.raises(NameError):
        fn.iir(signal, cutoff=s)
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
    with pytest.raises(NameError):
        fn.iir(signal, plot=yes)
    with pytest.raises(ValueError):
        fn.iir(signal, ftype='yes')
    with pytest.raises(NameError):
        fn.iir(signal, ftype=yes)
    with pytest.raises(TypeError):
        fn.iir(signal, ftype=12)
    with pytest.raises(ValueError):
        fn.iir(signal, ftype='12')
    with pytest.raises(TypeError):
        fn.iir(signal, filter=12)
    with pytest.raises(ValueError):
        fn.iir(signal, filter='yes')
    with pytest.raises(NameError):
        fn.iir(signal, filter=yes)
    with pytest.raises(ValueError):
        fn.iir(signal, filter='sa') 
    with pytest.raises(TypeError):
        fn.iir(signal, dummy=1)


def test_fir():
    with pytest.raises(ValueError):
        fn.fir(signal=1)
    with pytest.raises(TypeError):
        fn.fir(signal='sa')
    with pytest.raises(NameError):
        fn.fir(signal=sa)
    with pytest.raises(TypeError):
        fn.fir(signal=1+3j)
    with pytest.raises(ValueError):
        fn.fir(signal=[1, 1+3j])
    signal = fn.sinewave()
    with pytest.raises(ValueError):
        fn.fir(signal, fs='s')
    with pytest.raises(NameError):
        fn.fir(signal, fs=s)
    with pytest.raises(TypeError):
        fn.fir(signal, fs=[1, 2])
    with pytest.raises(TypeError):
        fn.fir(signal, fs=1+3j)
    with pytest.raises(ValueError):
        fn.fir(signal, ordern=-1)
    with pytest.raises(NameError):
        fn.fir(signal, ordern=s)
    with pytest.raises(ValueError):
        fn.fir(signal, cutoff=[1, 2, 3])
    with pytest.raises(TypeError):
        fn.fir(signal, cutoff=[1+3j, 4])
    with pytest.raises(NameError):
        fn.fir(signal, cutoff=s)
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
    with pytest.raises(NameError):
        fn.fir(signal, plot=yes)
    with pytest.raises(ValueError):
        fn.fir(signal, ftype='yes')
    with pytest.raises(NameError):
        fn.fir(signal, ftype=yes)
    with pytest.raises(TypeError):
        fn.fir(signal, ftype=12)
    with pytest.raises(ValueError):
        fn.fir(signal, ftype='12') 
    with pytest.raises(TypeError):
        fn.fir(signal, dummy=1)


def test_movavg():
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
    with pytest.raises(NameError):
        fn.movavg(signal=sa)
    with pytest.raises(TypeError):
        fn.movavg(signal=1+3j)
    with pytest.raises(ValueError):
        fn.movavg(signal=[1, 1+3j])
    signal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        fn.movavg(signal, -1)
    with pytest.raises(ValueError):
        fn.movavg(signal, 'a')
    with pytest.raises(NameError):
        fn.movavg(signal, a)
    with pytest.raises(TypeError):
        fn.movavg(signal, 1+3j)    


def test_polyfit():
    time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    signal = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    para = np.polyfit(time,signal,2)
    func = np.poly1d(para)
    reg = func(time)
    reg1 = fn.polyfit(time, signal, 2, plot='n')
    assert np.array_equal(reg, reg1)
    with pytest.raises(TypeError):
        fn.polyfit(time, signal, plot='happy')
    with pytest.raises(TypeError):
        fn.polyfit(time, signal, plot=1223)
    with pytest.raises(NameError):
        fn.polyfit(time, signal, plot=yes)
    with pytest.raises(ValueError):
        fn.polyfit(time, signal, degree=-2)
    with pytest.raises(ValueError):
        fn.polyfit(time=time, signal=1, degree=2)
    with pytest.raises(TypeError):
        fn.polyfit(time=time, signal='sa', degree=2)
    with pytest.raises(NameError):
        fn.polyfit(time=time, signal=sa, degree=2)
    with pytest.raises(TypeError):
        fn.polyfit(time=time, signal=1+3j, degree=2)
    with pytest.raises(ValueError):
        fn.polyfit(time=time, signal=[1, 1+3j], degree= 2)
    signal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        fn.polyfit(time=1,signal=signal, degree=2)
    with pytest.raises(TypeError):
        fn.polyfit(time='sa', signal=signal, degree=2)
    with pytest.raises(NameError):
        fn.polyfit(time=sa, signal=signal, degree=2)
    with pytest.raises(TypeError):
        fn.polyfit(time=1+3j, signal=signal, degree=2)
    with pytest.raises(ValueError):
        fn.polyfit(time=[1, 1+3j], signal=signal, degree=2)    


def test_xcorr():
    sig1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sig2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        fn.xcorr(sig1, sig2=1)
    with pytest.raises(TypeError):
        fn.xcorr(sig1, sig2='sa')
    with pytest.raises(NameError):
        fn.xcorr(sig1, sig2=sa)
    with pytest.raises(TypeError):
        fn.xcorr(sig1, sig2=1+3j)
    with pytest.raises(ValueError):
        fn.xcorr(sig1, sig2=[1, 1+3j])
    with pytest.raises(ValueError):
        fn.xcorr(sig1=1, sig2=sig2)
    with pytest.raises(TypeError):
        fn.xcorr(sig1='sa', sig2=sig2)
    with pytest.raises(NameError):
        fn.xcorr(sig1=sa, sig2=sig2)
    with pytest.raises(TypeError):
        fn.xcorr(sig1=1+3j, sig2=sig2)
    with pytest.raises(ValueError):
        fn.xcorr(sig1=[1, 1+3j], sig2=sig2)


def test_acorr():
    with pytest.raises(ValueError):
        fn.acorr(signal=1)
    with pytest.raises(TypeError):
        fn.acorr(signal='sa')
    with pytest.raises(NameError):
        fn.acorr(signal=sa)
    with pytest.raises(TypeError):
        fn.acorr(signal=1+3j)
    with pytest.raises(ValueError):
        fn.acorr(signal=[1, 1+3j])


def test_envelope():
    return None

def test_psd():
    return None

def test_rectify():
    return None

def test_rms():
    return None

def test_rms_sig():
    return None
