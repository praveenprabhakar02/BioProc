"""
test_emg.py

Author: Praveen Prabhakar KR
Email: praveenp@msu.edu

Module contains the tests for emg module.
"""

import pytest
import numpy as np
from .. import emg as em

def test_emgsig():
    """
    Test for emgsig function in the emg module.
    """

    with pytest.raises(TypeError):
        em.emgsig(seed='s')
    with pytest.raises(ValueError):
        em.emgsig(seed=-1)
    with pytest.raises(TypeError):
        em.emgsig(seed=1+3j)
    with pytest.raises(TypeError):
        em.emgsig(seed=[1+3j])
    with pytest.raises(ValueError):
        em.emgsig(plot='sa')
    with pytest.raises(TypeError):
        em.emgsig(plot=123)
    with pytest.raises(ValueError):
        em.emgsig(tarr='sa')
    with pytest.raises(TypeError):
        em.emgsig(tarr=123)
    with pytest.raises(TypeError):
        em.emgsig(tarr=1+3j)
    with pytest.raises(TypeError):
        em.emgsig(plot=1+3j)


def test_find_onset():
    """
    Test for find_onset function in the emg module.
    """

    with pytest.raises(ValueError):
        em.find_onset(signal=1)
    with pytest.raises(TypeError):
        em.find_onset(signal='sa')
    with pytest.raises(TypeError):
        em.find_onset(signal=1+3j)
    with pytest.raises(ValueError):
        em.find_onset(signal=[1, 1+3j])
    signal = em.emgsig()
    with pytest.raises(ValueError):
        em.find_onset(signal, fs='s')
    with pytest.raises(TypeError):
        em.find_onset(signal, fs=[1, 2])
    with pytest.raises(TypeError):
        em.find_onset(signal, fs=1+3j)
    with pytest.raises(TypeError):
        em.find_onset(signal, fs=[1+3j])
    with pytest.raises(ValueError):
        em.find_onset(signal, plot='sa')
    with pytest.raises(TypeError):
        em.find_onset(signal, plot=123)
    with pytest.raises(TypeError):
        em.find_onset(signal, plot=1+3j)
    with pytest.raises(TypeError):
        em.find_onset(signal, plot=[1+3j])
    with pytest.raises(ValueError):
        em.find_onset(signal, filt='sa')
    with pytest.raises(TypeError):
        em.find_onset(signal, filt=123)
    with pytest.raises(TypeError):
        em.find_onset(signal, filt=1+3j)
    with pytest.raises(TypeError):
        em.find_onset(signal, filt=[1+3j])


def test_norm_emg():
    """
    Test for norm_emg function in the emg module.
    """

    sig = em.emgsig()
    with pytest.raises(ValueError):
        em.norm_emg(signal=sig, mvic=1, plot='N')
    with pytest.raises(TypeError):
        em.norm_emg(signal=sig, mvic='sa', plot='N')
    with pytest.raises(TypeError):
        em.norm_emg(signal=sig, mvic=1+3j, plot='N')
    with pytest.raises(ValueError):
        em.norm_emg(signal=sig, mvic=[1, 1+3j], plot='N')
    mvic = em.emgsig(seed=10)
    with pytest.raises(ValueError):
        em.norm_emg(signal=1, mvic=mvic, plot='N')
    with pytest.raises(TypeError):
        em.norm_emg(signal='sa', mvic=mvic, plot='N')
    with pytest.raises(TypeError):
        em.norm_emg(signal=1+3j, mvic=mvic, plot='N')
    with pytest.raises(ValueError):
        em.norm_emg(signal=[1, 1+3j], mvic=mvic, plot='N')
    sig = em.emgsig()
    mvic = em.emgsig(seed=10)
    with pytest.raises(ValueError):
        em.norm_emg(signal=sig, fs='s', mvic=mvic, plot='N')
    with pytest.raises(TypeError):
        em.norm_emg(signal=sig, fs=[1, 2], mvic=mvic, plot='N')
    with pytest.raises(TypeError):
        em.norm_emg(signal=sig, fs=1+3j, mvic=mvic, plot='N')
    with pytest.raises(TypeError):
        em.norm_emg(signal=sig, fs=[1+3j], mvic=mvic, plot='N')
    with pytest.raises(ValueError):
        em.norm_emg(signal=sig, mvic=mvic, plot='sa')
    with pytest.raises(TypeError):
        em.norm_emg(signal=sig, mvic=mvic, plot=123)
    with pytest.raises(TypeError):
        em.norm_emg(signal=sig, mvic=mvic, plot=1+3j)
    with pytest.raises(TypeError):
        em.norm_emg(signal=sig, mvic=mvic, plot=[1+3j])
    with pytest.raises(ValueError):
        em.norm_emg(signal=sig, filt='sa', mvic=mvic, plot='N')
    with pytest.raises(TypeError):
        em.norm_emg(signal=sig, filt=123, mvic=mvic, plot='N')
    with pytest.raises(TypeError):
        em.norm_emg(signal=sig, filt=1+3j, mvic=mvic, plot='N')
    with pytest.raises(TypeError):
        em.norm_emg(signal=sig, filt=[1+3j], mvic=mvic, plot='N')


def test_padding():
    """
    Test for padding function in the emg module.
    """

    with pytest.raises(ValueError):
        em.padding(signal=1)
    with pytest.raises(ValueError):
        em.padding(signal=-1)
    with pytest.raises(TypeError):
        em.padding(signal='sa')
    with pytest.raises(TypeError):
        em.padding(signal=1+3j)
    with pytest.raises(ValueError):
        em.padding(signal=[1+3j])
    abab = [1, 2, 3, 4]
    baba = np.array([1, 2, 3, 4, 0, 0, 0, 0])
    assert np.array_equal(em.padding(abab), baba)
    abab = np.zeros(512)
    baba = em.padding(abab)
    assert baba.size == 1024
    abab = np.zeros(768)
    baba = em.padding(abab)
    assert baba.size == 1024
    abab = np.zeros(769)
    baba = em.padding(abab)
    assert baba.size == 2048


def test_emg_process():
    """
    Test for emg_process function in the emg module.
    """

    with pytest.raises(ValueError):
        em.emg_process(emg_signal=1, plot='N', fourier='N')
    with pytest.raises(ValueError):
        em.emg_process(emg_signal=-1, plot='N', fourier='N')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal='sa', plot='N', fourier='N')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=1+3j, plot='N', fourier='N')
    with pytest.raises(ValueError):
        em.emg_process(emg_signal=[1+3j], plot='N', fourier='N')
    sig = em.emgsig(seed=10)
    with pytest.raises(ValueError):
        em.emg_process(emg_signal=sig, mvic_signal=1, plot='N', fourier='N')
    with pytest.raises(ValueError):
        em.emg_process(emg_signal=sig, mvic_signal=-1, plot='N', fourier='N')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=sig, mvic_signal='sa', plot='N', fourier='N')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=sig, mvic_signal=1+3j, plot='N', fourier='N')
    with pytest.raises(ValueError):
        em.emg_process(emg_signal=sig, mvic_signal=[1+3j], plot='N', fourier='N')
    sig = em.emgsig()
    mvic = em.emgsig(seed=10)
    with pytest.raises(ValueError):
        em.emg_process(emg_signal=sig, fs='s', mvic=mvic, plot='N', fourier='N')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=sig, fs=[1, 2], mvic=mvic, plot='N', fourier='N')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=sig, fs=1+3j, mvic=mvic, plot='N', fourier='N')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=sig, fs=[1+3j], mvic=mvic, plot='N', fourier='N')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=sig, mvic=mvic, plot='Nasasd', fourier='N')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=sig, mvic=mvic, plot=123, fourier='N')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=sig, mvic=mvic, plot=1+3j, fourier='N')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=sig, mvic=mvic, plot='N', fourier='Nasas')
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=sig, mvic=mvic, plot='N', fourier=123)
    with pytest.raises(TypeError):
        em.emg_process(emg_signal=sig, mvic=mvic, plot='N', fourier=1+3j)
