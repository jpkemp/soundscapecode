from operator import add, sub
import numpy as np
from math import log10, floor
from scipy.signal import lfilter, kaiserord, firwin, freqz

class UnknownFilterType(ValueError):
    def __init__(self, filter_type, *args):
        msg = f"Filter must be lowpass, bandpass, or highpass, not {filter_type}"
        super().__init__(msg, *args)

def _db_voltage(x):
    '''Calculate power

    :meta private:
    '''
    power = abs(x**2)
    return 10*log10(power)

def _stop_calc(h, a_stop, idx):
    '''Returns true if stopband attenuation is within limits

    :meta private:
    '''
    # Measure attenuation defined as the distance between the nominal
    # gain(0 dB in our case) and the maximum rippple in the stopband.
    ngain = 1
    measAstop = _db_voltage(ngain)-_db_voltage(max(h))
    if measAstop <= a_stop[idx]:
        return False

    return True

def _pass_calc(h, a_pass, idx):
    '''Returns true if passband attenuation is within limits

    :meta private:
    '''
    # The ripple is defined as the amplitude (dB) variation between the two
    # specified frequency points.
    measApass = _db_voltage(max(h))-_db_voltage(min(h))
    if (measApass >= a_pass[idx]):
        return False

    return True

def _get_stop_bands(stopbands, filter_type, nyquist):
    '''Get full stop bands

    :meta private:
    '''
    if filter_type == "lowpass":
        return [(stopbands[0], nyquist)]
    elif filter_type == "bandpass":
        return [(0, stopbands[0]), (stopbands[1], nyquist)]
    elif filter_type == "highpass":
        return [(0, stopbands[0])]
    else:
        raise UnknownFilterType(filter_type)

def _get_passbands(passbands, filter_type, nyquist):
    '''Get full pass bands

    :meta private:
    '''
    if filter_type == "lowpass":
        return [(0, passbands[0])]
    elif filter_type == "bandpass":
        return [(passbands[0], passbands[1])]
    elif filter_type == "highpass":
        return [passbands[0], nyquist]
    else:
        raise UnknownFilterType(filter_type)

def _check_kaiser_specs(b, stopbands, passbands, fs, a_stop, a_pass, filter_type):
    '''Check if a kaiser filter is within spec

    :meta private:
    '''
    nyquist = fs / 2
    stopbands = _get_stop_bands(stopbands, filter_type, nyquist)
    passbands = _get_passbands(passbands, filter_type, nyquist)
    for bands, N, a, func in [(stopbands, 2**12, a_stop, _stop_calc), 
                            (passbands, 2**10, a_pass, _pass_calc)]:
        normalised_bands = [tuple(y / nyquist for y in x) for x in bands]
        for idx, (f_start, f_end) in enumerate(normalised_bands):
            linN = np.linspace(f_start, f_end, N)
            w, h = freqz(b, worN=linN, fs=2) # fs always 2 because all values are normalised
            h = abs(h)
            result = func(h, a, idx)
            if not result: return False

    return True
            
def _calc_w_stop(passband, steepness, fs, op=sub):
    '''calculate stopband

    :meta private:
    '''
    if steepness < 0.5 or steepness > 1:
        raise ValueError("Steepness must be between 0.5 and 1")

    nyquist = fs / 2
    TwPercentage = -0.98*steepness + 0.99
    WpassNormalized = passband/(nyquist)
    Tw = TwPercentage * WpassNormalized if op is sub else TwPercentage * (1 - WpassNormalized)
    WstopNormalized = op(WpassNormalized, Tw)
    Wstop = WstopNormalized * (nyquist)

    return Wstop

def _get_filter_ops(filter_type):
    '''get filter operator for calculation

    :meta private:
    '''
    if filter_type == "lowpass":
        return [add]
    elif filter_type == "highpass":
        return [sub]
    elif filter_type == "bandpass":
        return [sub, add]
    else:
        raise ValueError(f"filter_type must be one of lowpass, highpass, bandpass. got {filter_type}")

def _check_numtaps(numtaps, filter_type):
    '''Return odd taps for highpass and bandpass filters

    :meta private:
    '''
    if filter_type in ["highpass", "bandpass"]:
        numtaps |= 1

    return numtaps

def _get_valid_kaiser_filter_window(passbands, fs, stopband_atten, steepness=0.85, filter_type='bandpass'):
    '''Get a kaiser filter window which meets the spec

    :meta private:
    '''
    a_stop = 60
    a_pass = 0.1
    ops = _get_filter_ops(filter_type)
    stopbands = []
    cutoffs = []
    widths = []
    for i, band in enumerate(passbands):
        w_stop = _calc_w_stop(band, steepness, fs, ops[i])
        stopbands.append(w_stop)
        widths.append(abs(band - w_stop))
    
    width = min(widths)
    for i, band in enumerate(passbands):
        cutoffs.append(ops[i](band, width / 2))

    nyquist = fs / 2
    numtaps, beta = kaiserord(stopband_atten, width/(nyquist))
    numtaps = _check_numtaps(numtaps, filter_type)
    original_design_taps = firwin(numtaps, cutoffs, window=('kaiser', beta), scale=True, fs=fs, pass_zero=filter_type)
    # original_design_taps = firwin(numtaps, cutoffs, width=width, scale=True, fs=fs, pass_zero=filter_type) # gives slightly different results which cause enough differences to fail the invertebrate tests, as the original design spec passes
    valid = _check_kaiser_specs(original_design_taps, stopbands, passbands, fs, [a_stop] * len(cutoffs), [a_pass] * len(cutoffs), filter_type)
    if valid:
        return original_design_taps

    count = 1
    while not valid:
        numtaps += 1
        numtaps = _check_numtaps(numtaps, filter_type)
        taps = firwin(numtaps, cutoffs, window=('kaiser', beta), scale=True, fs=fs, pass_zero=filter_type)
        valid = _check_kaiser_specs(taps, stopbands, passbands, fs, [a_stop] * len(cutoffs), [a_pass] * len(cutoffs), filter_type)
        count += 1
        if count == 10:
            return original_design_taps

    return taps
    
def _filter(data, taps):
    ''' Filters the data with given taps

    :meta private:
    '''
    numtaps = len(taps)
    delay = floor(numtaps / 2)
    temp_data = np.concatenate([data, np.zeros(delay)])
    fltrd = lfilter(taps, 1, temp_data)[delay:]

    return fltrd

def highpass(data:np.ndarray, passband:int, fs:int, steepness=0.85, stopband_atten=60):
    '''Filter data with a highpass filter

    Parameters
    ----------
    data: np.ndarray
        The data to filter. Expected shape is (n,) where n is the length of the data
    
    passband: int
        The pass band for the filter

    fs: int
        Sampling frequency for the data

    steepness: float
        Steepness parameter for the width calculation. Defaults to 0.85.

    stopband_atten: int
        Acceptable stopband attenuation in dB. Defaults to 60 dB.

    Returns
    -------
    np.ndarray
        The filtered data.
    '''
    taps = _get_valid_kaiser_filter_window([passband], fs, stopband_atten, steepness, filter_type="highpass")

    return _filter(data, taps)

def bandpass(data:np.ndarray, band:tuple, fs, steepness=0.85, stopband_atten=60):
    '''Filter data with a bandpass filter

    Parameters
    ----------
    data: np.ndarray
        The data to filter. Expected shape is (n,) where n is the length of the data
    
    passband: tuple
        The pass band for the filter, in the form (start, end).

    fs: int
        Sampling frequency for the data

    steepness: float
        Steepness parameter for the width calculation. Defaults to 0.85.

    stopband_atten: int
        Acceptable stopband attenuation in dB. Defaults to 60 dB.

    Returns
    -------
    np.ndarray
        The filtered data.
    '''
    taps = _get_valid_kaiser_filter_window(band, fs, stopband_atten, steepness, filter_type="bandpass")

    return _filter(data, taps)
