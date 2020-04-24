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
    signal = em.emgsig()
    with pytest.raises(ValueError):
        em.emgsig(signal, plot='sa')
    with pytest.raises(TypeError):
        em.emgsig(signal, plot=123)
    with pytest.raises(NameError):
        em.emgsig(signal, plot=s)
    with pytest.raises(ValueError):
        em.emgsig(signal, tarr='sa')
    with pytest.raises(TypeError):
        em.emgsig(signal, tarr=123)
    with pytest.raises(NameError):
        em.emgsig(signal, tarr=s)
    with pytest.raises(TypeError):
        em.emgsig(signal, filt=1+3j)
    with pytest.raises(TypeError):
        em.emgsig(signal, plot=1+3j)


def test_norm_emg():
    return None


def test_padding():
    return None


def test_emg_process():
    return None