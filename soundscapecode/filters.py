from operator import add, sub
import numpy as np
from math import log10, sqrt, floor
from scipy.signal import lfilter, filtfilt, butter, kaiserord, firwin, freqz

def butter_highpass_coeffs(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    return b, a

def filter_data(data, cutoff, fs, func):
    b, a = func(cutoff, fs)
    y = filtfilt(b, a, data)

    return y

class UnknownBandError(ValueError):
    def __init__(self, band, source, target, *args):
        msg = f"Unknown band {band} for source {source} and target {target}"
        super().__init__(msg, *args)

class UnknownTargetError(ValueError):
    def __init__(self, source, target, *args):
        msg = f"Unknown source {source} for target {target}"
        super().__init__(msg, *args)

class UnknownSourceError(ValueError):
    def __init__(self, source, *args):
        msg = f"Unknown source {source}"
        super().__init__(msg, *args)

def convert_mag_units(val, source, target, band):
    if source == 'linear':
        if target == 'linear':
            if band == 'pass':
                return val
            if band == 'stop' or band == 'amplitude':
                return abs(val)
            raise UnknownBandError(band, source, target)
        if target == 'db':
            if band == 'pass':
                return 20 * log10((1 + val) / (1 - val))
            if band == 'stop':
                return -20 * log10(val)
            if band == 'amplitude':
                return 20 * log10(abs(val))
            raise UnknownBandError(band, source, target)
        if target == 'squared':
            if band == 'pass':
                return ((1 - val) / (1 + val))**2
            if band == 'stop':
                return val**2 # should be vector
            if band == 'amplitude':
                return val**2 # should be element-wise
            raise UnknownBandError(band, source, target)

        raise UnknownTargetError(source, target)
    if source == 'squared':
        if target == 'squared':
            return val
        if target == 'db':
            if band in ['band', 'stop']:
                return 10 * log10(1 / val)
            raise ValueError(f"Unknown band: {band}")
        if target == 'linear':
            if band == 'pass':
                return (1 - sqrt(val)) / (1 + sqrt(val))
            if band == 'stop':
                return sqrt(val)
            raise UnknownBandError(band, source, target)
        raise UnknownTargetError(source, target)
    if source == 'db':
        if target == 'db':
            return val
        if target == 'squared':
            if band in ['pass', 'stop']:
                return 1 / (10**(val / 10))
            raise UnknownBandError(band, source, target)
        if target == 'linear':
            if band == 'pass':
                return (10**(val/20) - 1) / (10**(val/20) + 1)
            if band == 'stop':
                return 10**(-val / 20) # vector
            if band == 'amplitude':
                return 10**(val / 20) # element-wise

        raise UnknownTargetError(source, target)

    raise UnknownSourceError(source)

def check_kaiser_specs(b, stopbands, passbands, fs):
    N = 2**12
    n_points = len(stopbands)
    for idx in range(1, n_points):
        Fstart = stopbands[idx - 1] if idx else 0
        Fend = stopbands(idx) if idx else fs
        #Get fft at desired bands, we always get normalized frequency values in
        #hspecs so set Fs parameter to 2
        linN = np.linspace(Fstart, Fend, N)
        h = abs(freqz(b, worN=N, fs=fs))
        
        # Measure attenuation defined as the distance between the nominal
        # gain(0 dB in our case) and the maximum rippple in the stopband.
        ngain = 1
        measAstop = db(ngain)-db(max(h))
        if (measAstop <= Astop(idx))
            return #return with status = false
                #Measure ripple at the passbands
    N = 2**10
    for idx = 1:size(passbands,1)
        Fstart = passbands(idx,1)
        Fend = passbands(idx,2)
        #Get fft at desired bands, we always get normalized frequency values in
        #hspecs so set Fs parameter to 2    
        h = abs(freqz(b, worN=N, fs))
        # The ripple is defined as the amplitude (dB) variation between the two
        # specified frequency points.
        measApass = db(max(h))-db(min(h))
        if (measApass >= Apass(idx))
            return #return with status = false

def highpass(data, passband:tuple, fs:int, steepness=0.85, stopband_atten=60):
    if steepness < 0.5 or steepness > 1:
        raise ValueError("Steepness must be between 0.5 and 1")

    TwPercentage = -0.98*steepness + 0.99
    # ripple = 0.1
    WpassNormalized = passband/(fs/2)
    Tw =TwPercentage * WpassNormalized
    WstopNormalized = WpassNormalized - Tw
    # stopband_atten_linear = convert_mag_units(stopband_atten, 'db', 'linear', 'stop')
    # passband_ripple_linear = convert_mag_units(ripple, 'db', 'linear', 'pass')
    Wstop = WstopNormalized * (fs / 2)
    trans_width = passband - Wstop

    numtaps, beta = kaiserord(stopband_atten, trans_width/(0.5*fs))
    cutoff = passband - (trans_width / 2)
    taps = firwin(numtaps, cutoff, width=trans_width, scale=True, fs=fs, pass_zero='highpass')
    # taps = firwin(numtaps, mid_freq, window=('kaiser', beta), scale=True, fs=fs, pass_zero='highpass')
    # taps = firwin(numtaps, mid_freq / (fs/2), window=('kaiser', beta), scale=True, fs=fs, pass_zero='highpass')
    delay = floor(numtaps / 2)
    temp_data = np.concatenate([data, np.zeros(delay)])
    fltrd = lfilter(taps, 1, temp_data)[delay:]

    return fltrd

def bandpass(data, low, high, fs, steepness=0.85, stopband_atten=60):
    if steepness < 0.5 or steepness > 1:
        raise ValueError("Steepness must be between 0.5 and 1")

    nyquist = fs / 2
    TwPercentage = -0.98*steepness + 0.99
    # ripple = 0.1
    trans_widths = []
    cutoffs = []
    for passband, op in [(low, sub), (high, add)]:
        WpassNormalized = passband/(nyquist)
        Tw = TwPercentage * WpassNormalized if op is sub else TwPercentage * (1 - WpassNormalized)
        WstopNormalized = op(WpassNormalized, Tw)
        Wstop = WstopNormalized * (nyquist)
        trans_width = abs(passband - Wstop)
        trans_widths.append(trans_width)
        mid_freq = op(passband, trans_width / 2) # unclear whether this should be min trans_width or relative
        cutoffs.append(mid_freq)

    trans_width = min(trans_widths)
    numtaps, beta = kaiserord(stopband_atten, trans_width/(nyquist))
    taps = firwin(numtaps, cutoffs, width=trans_width, scale=True, fs=fs, pass_zero='bandpass')
    # taps = firwin(numtaps, mid_freqs, window=('kaiser', beta), scale=True, fs=fs, pass_zero='bandpass')
    delay = floor(numtaps / 2)
    temp_data = np.concatenate([data, np.zeros(delay)])
    fltrd = lfilter(taps, 1, temp_data)[delay:]

    return fltrd