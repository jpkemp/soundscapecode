import numpy as np
import scipy.fft as sp_fft
from math import log10, sqrt
from scipy.signal import find_peaks
from scipy.stats import kurtosis as spkurtosis

def _hilbert(data, axis=0):
    x = np.asarray(data)
    N = x.shape[axis]
    Xf = sp_fft.fft(x, N, axis=axis)
    h = np.zeros(N, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]

    x = sp_fft.ifft(Xf * h, axis=axis)

    return x
def _ensure_np(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if len(data.shape) > 1 and data.shape[1] != 1:
        raise AttributeError("data must be a vector")

    return data

def _mean_point_1(data:np.ndarray, fs):
    mean_spl = []
    interval = int(fs * 0.1)
    for i in range(0, len(data), interval):
        x = data[i:i + interval].mean()
        mean_spl.append(x)

    return np.array(mean_spl)

def _diff(data_a:np.ndarray, data_b:np.ndarray):
    ret = []
    for i in range(1, data_a.size):
        ret.append(data_b[i] - data_a[i])

    return ret

def temporal_dissimilarity(data_a:np.ndarray, data_b:np.ndarray):
    '''
    Calculates the temporal dissimilarity between two sounds.

    data_b: an array-like (numpy ndarray expected)
    data_b: an array-like (numpy ndarray expected)

    returns: the RMS SPL
    '''
    datas = []
    for data in (data_a, data_b):
        datas.append(_ensure_np(data))

    if data.size != data_b.size:
        raise AttributeError("Sounds must be the same size to calculate temporal dissimilarity")

    compare = []
    for data in [data_a, data_b]:
        transformed = _hilbert(data)
        abs_t = np.abs(transformed)
        A = abs_t / abs_t.sum()
        compare.append(A)

    dt = np.abs(_diff(*compare)).sum() / 2

    return dt

def max_spl(data:np.ndarray, reference_sound_pressure=1)->float:
    '''
    Calculates the maximum instantaneous sound pressure level for sound data.

    data: an array-like (numpy ndarray expected)
    reference_sound_pressure: p_0 in uPa 

    returns: the RMS SPL
    '''
    data = _ensure_np(data)
    return 10 * log10((np.abs(data)**2).max() / reference_sound_pressure)

def rms_spl(data:np.ndarray, fs:int, reference_sound_pressure=1)->float:
    '''
    Calculates the root-mean-squared sound pressure level for sound data.

    data: a vector-like (numpy ndarray expected)
    fs: the sampling frequency
    reference_sound_pressure: p_0 in uPa 

    returns: the RMS SPL
    '''
    data = _ensure_np(data)
    squared_sum = (data ** 2).sum()
    return 20 * log10(sqrt(squared_sum / (reference_sound_pressure * fs * 60)))

def kurtosis(data:np.ndarray)->float:
    '''
    Calculates the kurtosis for sound data.

    data: a vector-like (numpy ndarray expected)
    
    returns: the kurtosois
    '''
    data = _ensure_np(data)
    B = spkurtosis(data, fisher=False)
    assert len(B) == 1

    return B[0]

def periodicity(data:np.ndarray, fs)->float:
    '''
    Calculates the periodicity for sound data.

    data: a vector-like (numpy ndarray expected)
    
    returns: the kurtosois
    '''
    data = _ensure_np(data)
    cs = _mean_point_1(data, fs)
    xc = np.correlate(cs, cs, "full")
    mid = int((len(xc) + 1) / 2) - 1
    xc /= xc[mid]
    peaks = find_peaks(xc, prominence=0.1)
    n_peaks = len(peaks[0])

    return n_peaks
