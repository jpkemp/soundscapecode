import numpy as np
import scipy.fft as sp_fft
from math import log10, sqrt
from scipy.signal import get_window, spectrogram, find_peaks
from scipy.stats import kurtosis as spkurtosis

def _hilbert(data, axis=0):
    '''Hilbert transform

    :meta private:
    '''
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
    '''Check vector

    :meta private:
    '''
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if len(data.shape) > 1 and data.shape[1] != 1:
        raise AttributeError("data must be a vector")
    elif len(data.shape) == 1:
        data = np.reshape(data, (-1, 1))

    return data

def _mean_point_1(data:np.ndarray, fs):
    '''Mean value at 0.1 s

    :meta private:
    '''
    mean_spl = []
    interval = int(fs * 0.1)
    for i in range(0, len(data), interval):
        x = data[i:i + interval].mean()
        mean_spl.append(x)

    return np.array(mean_spl)

def _diff(data_a:np.ndarray, data_b:np.ndarray):
    '''element-wise difference between arrays

    :meta private:
    '''
    ret = []
    for i in range(1, data_a.size):
        ret.append(data_b[i] - data_a[i])

    return ret

def meanfreq(pxx:np.ndarray, f:np.ndarray, freqrange:tuple[float]=None):
    '''Calculates the mean frequenc of a power spectral density estimate.

    Parameters
    ----------
    pxx: np.ndarray
        The power spectral density estimate.
    f: np.ndarray
        Vector of the frequencies in pxx.
    freqrange: tuple[float]
        two-tuple of (lower, upper) frequency bounds. Defaults to None, which includes all frequencies.

    Returns
    -------
    np.ndarray
        A vector containing the mean frequency at each time step in pxx.
    '''
    width = np.tile(f[1]-f[0], (1, pxx.shape[0])).T
    f = np.reshape(f, (-1, 1))
    P = pxx * width
    if freqrange:
        lower = next((idx for idx, freq in enumerate(f) if freq > freqrange[0]), 0) - 1
        upper = next((idx for idx, freq in enumerate(f) if freq > freqrange[1]), len(f))
    else:
        lower = 0
        upper = f.shape[0]

    pwr = np.sum(P[lower:upper], 0, keepdims=True).T

    mnfreq = np.dot(P[lower:upper].T, f[lower:upper]) / pwr

    return np.squeeze(mnfreq)

def power_spectral_density(data:np.ndarray, fs:int):
    '''Estimates the power spectral density of a sound recording. Resulting PSD matches that calculated by the Matlab spectrogram function.

    Parameters
    ----------
    data:np.ndarray
        An array-like with shape (n, 1)
    fs:int
        The sampling frequency of the data

    Returns
    -------
    np.ndarray:
        frequency vector
    np.ndarray
        time vector
    np.ndarray
        psd matrix

    Raises
    ------
    AttributeError
        if the data input is not a vector
    '''
    data = _ensure_np(data)
    window_length = fs
    window = get_window('hamming', window_length)
    f, t, pxx = spectrogram(data, fs=fs, window=window, noverlap=window_length/2, mode='psd',
            scaling='density', nperseg=window_length, nfft=window_length, axis=0, detrend=False)

    pxx = np.squeeze(pxx)


    return f[1:], t, pxx[1:]

def temporal_dissimilarity(data_a:np.ndarray, data_b:np.ndarray)->float:
    '''Calculates the temporal dissimilarity between two sounds.

    Parameters
    ----------
    data_a: np.ndarray
        an array-like with shape (n, 1)
    data_b: np.ndarray
        an array-like with shape (n, 1)

    Returns
    -------
    float
        The temporal dissimilarity

    Raises
    ------
    AttributeError
        if either data input is not a vector
    AttributeError
        if the data input lengths are not the same

    Examples
    -----
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> import soundscapecode as ssc
    >>> fs = 48000
    >>> sound_a = np.random.rand(fs*60,1)
    >>> sound_b = np.random.rand(fs*60,1)
    >>> ssc.temporal_dissimilarity(sound_a, sound_b)
    0.2585992462980159
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

def spectral_dissimilarity(m_freq_a, m_freq_b):
    '''Calculates the spectral dissimilarity between two mean frequency inputs.

    Parameters
    ----------
    m_freq_a: np.ndarray
        an array-like with shape (n, 1)
    m_freq_b: np.ndarray
        an array-like with shape (n, 1)

    Returns
    -------
    float
        The spectral dissimilarity between the inputs

    Raises
    ------
    AttributeError
        if either data input is not a vector
    AttributeError
        if the data input lengths are not the same

    Examples
    -----
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> import soundscapecode as ssc
    >>> fs = 16000
    >>> sound = np.random.rand(fs*60*3,1)
    >>> f, t, pxx = ssc.power_spectral_density(sound, fs)
    >>> m_freq = ssc.meanfreq(pxx, f)
    >>> freq_a = m_freq[0:120] # half-second time steps
    >>> freq_b = m_freq[120:240]
    >>> ssc.spectral_dissimilarity(freq_a, freq_b)
    0.006182274981624808
    '''
    datas = []
    for mfreq in (m_freq_a, m_freq_b):
        tobs = np.abs(mfreq) / np.abs(mfreq).sum()
        datas.append(tobs)

    df = np.abs(_diff(*datas)).sum() / 2

    return df

def dissimilarity_index(data_a:np.ndarray,
                        data_b:np.ndarray,
                        m_freq_a:np.ndarray,
                        m_freq_b:np.ndarray)->list:
    '''Calculates the dissimilarity index between two sounds and the mean frequency inputs of the sounds.

    Parameters
    ----------
    data_a:np.ndarray
        an array-like with shape (n, 1)
    data_b:np.ndarray
        an array-like with shape (n, 1)
    m_freq_a:np.ndarray
        an array-like with shape (n, 1)
    m_freq_a:np.ndarray
        an array-like with shape (n, 1)

    Returns
    -------
    float
        The dissimilarity index, calculated from the temporal dissimilarity between data_a and data_b, and the spectral dissimilarity between m_freq_a and m_freq_b.

    Raises
    ------
    AttributeError
        if either data input is not a vector
    AttributeError
        if the data input lengths are not the same

    Examples
    -----
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> import soundscapecode as ssc
    >>> fs = 16000
    >>> sound = np.random.rand(fs*60*3,1)
    >>> data_a = sound[0:120]
    >>> data_b = sound[120:240]
    >>> f, t, pxx = ssc.power_spectral_density(sound, fs)
    >>> m_freq = ssc.meanfreq(pxx, f)
    >>> freq_a = m_freq[0:120] # half-second time steps
    >>> freq_b = m_freq[120:240]
    >>> ssc.dissimilarity_index(data_a, data_b, freq_a, freq_b)
    0.001438456599369862
    '''
    if data_a.shape != data_b.shape:
        raise AttributeError("Vectors must be the same size")

    datas = []
    for data in [data_a, data_b]:
        data = _ensure_np(data)
        datas.append(data)

    freqs = []
    for freq in [m_freq_a, m_freq_b]:
        freq = _ensure_np(freq)
        freqs.append(freq)

    dt = temporal_dissimilarity(*datas)
    df = spectral_dissimilarity(*freqs)

    return dt * df

def max_spl(data:np.ndarray, reference_sound_pressure:int=1)->float:
    '''Calculates the maximum instantaneous sound pressure level for sound data.

    Parameters
    ----------
    data: np.ndarray
        an array-like with shape (n, 1)
    reference_sound_pressure: int
        p_0 in uPa, defaults to 1

    Returns
    -------
    float
        The maximum instantaneous SPL

    Raises
    ------
    AttributeError:
        if either data input is not a vector

    Examples
    -----
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> import soundscapecode as ssc
    >>> fs = 48000
    >>> sound = np.random.rand(fs*60,1)
    >>> ssc.max_spl(sound, fs)
    -46.8124147991551
    '''
    data = _ensure_np(data)
    return 10 * log10((np.abs(data)**2).max() / reference_sound_pressure)

def rms_spl(data:np.ndarray, fs:int, reference_sound_pressure:int=1)->float:
    '''Calculates the root-mean-squared sound pressure level for sound data.

    Parameters
    ----------
    data: np.ndarray
        an array-like with shape (n, 1)
    fs: int
        the sampling frequency
    reference_sound_pressure:int
        p_0 in uPa, defaults to 1

    Returns
    -------
    float
        the RMS SPL

    Raises
    ------
    AttributeError
        if either data input is not a vector

    Examples
    -----
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> import soundscapecode as ssc
    >>> fs = 48000
    >>> sound = np.random.rand(fs*60,1)
    >>> ssc.rms_spl(sound, fs)
    -4.770500024074682
    '''
    data = _ensure_np(data)
    squared_sum = (data ** 2).sum()
    return 20 * log10(sqrt(squared_sum / (reference_sound_pressure * fs * 60)))

def kurtosis(data:np.ndarray)->float:
    '''Calculates the kurtosis for sound data.

    Parameters
    ----------
    data: np.ndarray
        an array-like with shape (n, 1)

    Returns
    -------
    float
        The kurtosois

    Raises
    ------
    AttributeError
        if either data input is not a vector

    Examples
    -----
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> import soundscapecode as ssc
    >>> fs = 48000
    >>> sound = np.random.rand(fs*60,1)
    >>> ssc.kurtosis(sound)
    1.799334080600504
    '''
    data = _ensure_np(data)
    B = spkurtosis(data, fisher=False)
    assert len(B) == 1

    return B[0]

def periodicity(data:np.ndarray, fs)->int:
    '''Calculates the periodicity for sound data.

    Parameters
    ----------
    data: np.ndarray
        an array-like with shape (n, 1)

    Returns
    -------
    int
        The periodicity

    Raises
    ------
    AttributeError
         if either data input is not a vector

    Examples
    -----
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> import soundscapecode as ssc
    >>> fs = 48000
    >>> sound = np.random.rand(fs*60,1)
    >>> ssc.periodicity(sound, fs)
    1
    '''
    data = _ensure_np(data)
    cs = _mean_point_1(data, fs)
    xc = np.correlate(cs, cs, "full")
    mid = int((len(xc) + 1) / 2) - 1
    xc /= xc[mid]
    peaks = find_peaks(xc, prominence=0.1)
    n_peaks = len(peaks[0])

    return n_peaks

if __name__ == "__main__":
    import doctest
    doctest.testmod()