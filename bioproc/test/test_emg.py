"""
test_emg.py
Author: Praveen Prabhakar KR
Email: praveenprabhakar02@gmail.com

Module contains the tests for emg module.
"""

import pytest
import numpy as np
from .. import emg as em

def test_emgsig():
    """
    Test for emgsig function in the emg module.
    """

    with pytest.raises(NameError):
        em.emgsig(seed=s)
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
    with pytest.raises(NameError):
        em.emgsig(plot=s)
    with pytest.raises(ValueError):
        em.emgsig(tarr='sa')
    with pytest.raises(TypeError):
        em.emgsig(tarr=123)
    with pytest.raises(NameError):
        em.emgsig(tarr=s)
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
    with pytest.raises(NameError):
        em.find_onset(signal=sa)
    with pytest.raises(TypeError):
        em.find_onset(signal=1+3j)
    with pytest.raises(ValueError):
        em.find_onset(signal=[1, 1+3j])
    signal = em.emgsig()
    with pytest.raises(ValueError):
        em.find_onset(signal, fs='s')
    with pytest.raises(NameError):
        em.find_onset(signal, fs=s)
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
    with pytest.raises(NameError):
        em.find_onset(signal, plot=s)
    with pytest.raises(TypeError):
        em.find_onset(signal, plot=1+3j)
    with pytest.raises(TypeError):
        em.find_onset(signal, plot=[1+3j])
    with pytest.raises(ValueError):
        em.find_onset(signal, filt='sa')
    with pytest.raises(TypeError):
        em.find_onset(signal, filt=123)
    with pytest.raises(NameError):
        em.find_onset(signal, filt=s)
    with pytest.raises(TypeError):
        em.find_onset(signal, filt=1+3j)
    with pytest.raises(TypeError):
        em.find_onset(signal, filt=[1+3j])


def test_norm_emg():
    """
    Test for norm_emg function in the emg module.
    """

    
    return None


def test_padding():
    """
    Test for padding function in the emg module.
    """

    
    return None


def test_emg_process():
    """
    Test for emg_process function in the emg module.
    """

    
    return None