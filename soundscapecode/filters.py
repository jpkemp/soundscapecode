from operator import add, sub, ge, le
import numpy as np
from math import log10, sqrt, floor
from numbers import Number
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

class UnknownFilterType(ValueError):
    def __init__(self, filter_type, *args):
        msg = f"Filter must be lowpass, bandpass, or highpass, not {filter_type}"
        super().__init__(msg, *args)

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

def db_voltage(x):
    power = abs(x**2)
    return 10*log10(power)

def stop_calc(h, a_stop, idx):
    # Measure attenuation defined as the distance between the nominal
    # gain(0 dB in our case) and the maximum rippple in the stopband.
    ngain = 1
    measAstop = db_voltage(ngain)-db_voltage(max(h))
    if measAstop <= a_stop[idx]:
        return False

    return True

def pass_calc(h, a_pass, idx):
    # The ripple is defined as the amplitude (dB) variation between the two
    # specified frequency points.
    measApass = db_voltage(max(h))-db_voltage(min(h))
    if (measApass >= a_pass[idx]):
        return False

    return True

def get_stop_bands(stopbands, filter_type, nyquist):
    if filter_type == "lowpass":
        return [(stopbands[0], nyquist)]
    elif filter_type == "bandpass":
        return [(0, stopbands[0]), (stopbands[1], nyquist)]
    elif filter_type == "highpass":
        return [(0, stopbands[0])]
    else:
        raise UnknownFilterType(filter_type)

def get_passbands(passbands, filter_type, nyquist):
    if filter_type == "lowpass":
        return [(0, passbands[0])]
    elif filter_type == "bandpass":
        return [(passbands[0], passbands[1])]
    elif filter_type == "highpass":
        return [passbands[0], nyquist]
    else:
        raise UnknownFilterType(filter_type)

def check_kaiser_specs(b, stopbands, passbands, fs, a_stop, a_pass, filter_type):
    nyquist = fs / 2
    stopbands = get_stop_bands(stopbands, filter_type, nyquist)
    passbands = get_passbands(passbands, filter_type, nyquist)
    for bands, N, a, func in [(stopbands, 2**12, a_stop, stop_calc), 
                            (passbands, 2**10, a_pass, pass_calc)]:
        normalised_bands = [tuple(y / nyquist for y in x) for x in bands]
        for idx, (f_start, f_end) in enumerate(normalised_bands):
            linN = np.linspace(f_start, f_end, N)
            w, h = freqz(b, worN=linN, fs=2) # fs always 2 because all values are normalised
            h = abs(h)
            result = func(h, a, idx)
            if not result: return False

    return True
            
def calc_w_stop(passband, steepness, fs, op=sub):
    if steepness < 0.5 or steepness > 1:
        raise ValueError("Steepness must be between 0.5 and 1")

    nyquist = fs / 2
    TwPercentage = -0.98*steepness + 0.99
    WpassNormalized = passband/(nyquist)
    Tw = TwPercentage * WpassNormalized if op is sub else TwPercentage * (1 - WpassNormalized)
    WstopNormalized = op(WpassNormalized, Tw)
    Wstop = WstopNormalized * (nyquist)

    return Wstop

def get_filter_ops(filter_type):
    if filter_type == "lowpass":
        return [add]
    elif filter_type == "highpass":
        return [sub]
    elif filter_type == "bandpass":
        return [sub, add]
    else:
        raise ValueError(f"filter_type must be one of lowpass, highpass, bandpass. got {filter_type}")

def check_numtaps(numtaps, filter_type):
    if filter_type in ["highpass", "bandpass"]:
        numtaps |= 1

    return numtaps

def get_valid_kaiser_filter_window(passbands, fs, stopband_atten, steepness=0.85, filter_type='bandpass'):
    a_stop = 60
    a_pass = 0.1
    ops = get_filter_ops(filter_type)
    stopbands = []
    cutoffs = []
    widths = []
    for i, band in enumerate(passbands):
        w_stop = calc_w_stop(band, steepness, fs, ops[i])
        stopbands.append(w_stop)
        widths.append(abs(band - w_stop))
    
    width = min(widths)
    for i, band in enumerate(passbands):
        cutoffs.append(ops[i](band, width / 2)) # should this be the min width or the current width? unclear

    nyquist = fs / 2
    numtaps, beta = kaiserord(stopband_atten, width/(nyquist))
    numtaps = check_numtaps(numtaps, filter_type)
    original_design_taps = firwin(numtaps, cutoffs, window=('kaiser', beta), scale=True, fs=fs, pass_zero=filter_type)
    valid = check_kaiser_specs(original_design_taps, stopbands, passbands, fs, [a_stop] * len(cutoffs), [a_pass] * len(cutoffs), filter_type)
    if valid:
        return original_design_taps

    count = 1
    while not valid:
        numtaps += 1
        numtaps = check_numtaps(numtaps, filter_type)
        taps = firwin(numtaps, cutoffs, window=('kaiser', beta), scale=True, fs=fs, pass_zero=filter_type)
        valid = check_kaiser_specs(taps, stopbands, passbands, fs, [a_stop] * len(cutoffs), [a_pass] * len(cutoffs), filter_type)
        count += 1
        if count == 10:
            return original_design_taps

    return taps
    
def highpass(data, passband:int, fs:int, steepness=0.85, stopband_atten=60):
    taps = get_valid_kaiser_filter_window([passband], fs, stopband_atten, steepness, filter_type="highpass")
    numtaps = len(taps)
    delay = floor(numtaps / 2)
    temp_data = np.concatenate([data, np.zeros(delay)])
    fltrd = lfilter(taps, 1, temp_data)[delay:]

    return fltrd

def bandpass(data, low, high, fs, steepness=0.85, stopband_atten=60):
    taps = get_valid_kaiser_filter_window([low, high], fs, stopband_atten, steepness, filter_type="bandpass")
    numtaps = len(taps)
    delay = floor(numtaps / 2)
    temp_data = np.concatenate([data, np.zeros(delay)])
    fltrd = lfilter(taps, 1, temp_data)[delay:]

    return fltrd