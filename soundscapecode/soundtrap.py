'''
Functions for working with soundtrap information

Functions
---------
read_calibration(file_path:str)->CalibrationData:
    Read soundtrap calibration information from file

get_soundtrap_calibration(serial:str, output_path:Path=None)->CalibrationData:
    'Retrieve soundtrap information from OceanInstruments and save to output path

soundtrap_conversion(signal:np.ndarray, soundtrap:str)->np.ndarray:
    Convert raw soundtrap values to uPa

open_wav(input_file:str, channel:int=None, trim_start:int=0, length:int=-1, soundtrap:int=None, normalise:bool=True)
    Open and convert a wave file signal to dB.

Classes
-------
CalibrationData
    Dataclass for soundtrap calibration information
'''
from dataclasses import dataclass
import json
import requests
import numpy as np
import wave
from pathlib import Path

@dataclass
class CalibrationData:
    '''Holds soundtrap calibration information

    Attributes
    ----------
    serial: str
        soundtrap serial no.
    device: str
        calibration device ID
    model: str
        soundtrap model no.
    date: str
        calibration date
    source: str
        tone source
    level: str
        source level in dB
    frequency: int
        source frequency in Hz
    RTI: float
        RTI level @ 1 kHz
    low: float
        low gain in dB
    high: float
        high gain in dB
    '''
    serial: str
    device: str
    model: str
    date: str
    source: str
    level: float
    frequency: int
    RTI: float
    low: float
    high: float

def _convert_json_to_cal(data)->CalibrationData:
    '''Converts OceanInstruments soundtrap information to CalibrationData

    Parameters
    ----------
    data:
        json sountrap information

    Returns
    -------
    CalibrationData:
        the soundtrap information object
    '''
    ret = CalibrationData(
        serial = data["Serial No"],
        device = data["Device ID"],
        model = data["Model"],
        date = data["Test Date"],
        source = data["Tone Source"],
        level = data["Source Level"],
        frequency = int(data["Frequency"].split('Hz')[0]),
        RTI = float(data["RTI Level @ 1kHz"].split('dB')[0]),
        low = float(data["Low gain"].split('dB')[0]),
        high = float(data["High gain"].split('dB')[0]),
    )

    return ret

def read_calibration(file_path:str)->CalibrationData:
    '''Read soundtrap calibration information from file

    Parameters
    ----------
    file_path:str
        the file with the sountrap information

    Returns
    -------
    CalibrationData:
        Calibration information from file

    Raises
    ------
    FileNotFoundError:
        the given path does not exist
    '''
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Cannot find file {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    return _convert_json_to_cal(data)

def get_soundtrap_calibration(serial:str, output_path:Path=None)->CalibrationData:
    '''Retrieve soundtrap information from OceanInstruments and save to output path

    Parameters
    ----------
    serial: int/str
        the soundtrap ID
    output_path: str/Path
        path to save the information to

    Returns
    -------
    CalibrationData:
        the soundtrap calibration information from OceanInstruments
    '''
    url = f"http://oceaninstruments.azurewebsites.net/api/Devices/Search/{serial}"
    device = requests.get(url)
    if device.status_code != 200:
        raise ValueError(f"Invalid response for device {serial}: {device.status_code}")

    device = device.json()
    device_id = device[0]["deviceId"]
    url = f"http://oceaninstruments.azurewebsites.net/api/Calibrations/Device/{device_id}"
    calibration = requests.get(url)
    if calibration.status_code != 200:
        raise ValueError(f"Invalid response for device {device_id}: {calibration.status_code}")

    calibration = calibration.json()


    tone = calibration[0]["tone"]
    level = calibration[0]["refLevel"]
    low = calibration[0]["lowFreq"]
    high = calibration[0]["highFreq"]
    source = calibration[0]["refDevice"]

    output = {
        "Serial No": serial,
        "Device ID": device_id,
        "Model": device[0]["modelName"],
        "Test Date": calibration[0]["dateCreated"],
        "Tone Source": source,
        "Source Level": f"{level} re. 1μPa",
        "Frequency": "250Hz",
        "RTI Level @ 1kHz": f"{tone}dB re. 1μPa",
        "Low gain": low if isinstance(low, str) else f"{low}dB",
        "High gain": high if isinstance(high, str) else f"{high}dB",
    }

    if output_path:
        with open(output_path, 'w+', encoding='utf-8') as f:
            f.write("{\n")
            for i, (key, val) in enumerate(output.items()):
                line = f"    \"{key}\": \"{val}\""
                if i != len(output) - 1:
                    line += ','

                line += '\n'
                f.write(line)

            f.write("}\n")

    if source != "CENTER 327":
       print("Warning: Frequency is unknown for this model and may not be correct. Suggest checking the Ocean Instruments calibration website")

    return _convert_json_to_cal(output)

def soundtrap_conversion(signal:np.ndarray, soundtrap:str)->np.ndarray:
    '''Convert raw soundtrap values to uPa

        Parameters
        ----------
        signal:np.ndarray
            signal to convert (iterable)
        sountrap:int/str
            soundtrap ID. if int, reads OceanInstruments website. if str, reads from file

        Returns
        -------
        converted signal
    '''
    if isinstance(soundtrap, int):
        cal_data = get_soundtrap_calibration(soundtrap)
    else:
        cal_data = read_calibration(soundtrap)
    cal_value = np.abs(cal_data.high)
    ratio = 10**(cal_value / 20)

    return signal * ratio

def _normalise_sound(data, data_format):
    '''Scales data to range [-1, 1] based on the wav format

    Parameters
    ---------
    data:np.ndarray
    data_format

    Returns
    -------
    np.ndarray
    '''
    def temp(x):
        if not x:
            return 0.0
        if x > 0:
            return float(x) /np.iinfo(data_format).max

        return float(x) / abs(np.iinfo(data_format).min)

    conv = np.vectorize(temp)
    return conv(data)

def open_wav(input_file:str, channel:int=None, trim_start:int=0, length:int=-1, soundtrap:int=None, normalise:bool=True)->tuple:
    ''' Open a wave file

        Parameters
        ----------
        input_file:
            file to open
        channel:int
            channel to use in a multi-channel file. Defaults to None, which uses channel 1
        trim_start: int
            cut the start of a file (in seconds)
        length: int
            length of the signal to keep (in seconds). Defaults to -1, which keeps the whole file
        soundtrap: int or str
            soundtrap ID for calibration. if int, retrieves the data from the OceanInstruments website. if str, loads the data from file

        Returns
        -------
        tuple
            (fs, signal) where fs is the sampling frequency and signal is the loaded sound
    '''
    wave_format = {1: np.uint8, 2: np.int16, 4:np.int32}
    with wave.open(str(input_file), 'rb') as file:
        nch = file.getnchannels()
        sampwidth = file.getsampwidth()
        frames = file.readframes(-1)
        nframes = file.getnframes()
        fs = file.getframerate()
        if sampwidth == 3:
            a = np.ndarray((nframes * nch * sampwidth), dtype=np.uint8, buffer=frames)
            b = np.empty((nframes, nch, 4), dtype=np.uint8)
            b[:, :, :sampwidth] = a.reshape(-1, nch, sampwidth)
            b[:, :, :sampwidth] = (b[:, :, sampwidth - 1:sampwidth] >> 7) * 255
            a = b.view('<i4').reshape(b.shape[:-1])
        else:
            a = np.ndarray((nframes * nch,), dtype=wave_format[sampwidth], buffer=frames)

        if normalise:
            a = _normalise_sound(a, wave_format[sampwidth])

    if nch > 1:
        if channel is None:
            print("Warning: no channel set for a multi-channel file. Using channel 1")
            channel = 1
        channels = [ [] for _ in range(nch) ]
        for index, value in enumerate(a):
            channels[index % nch].append(value)

        signal = np.array(channels[channel - 1])
    else:
        if channel is not None and channel != 1:
            print("Warning: invalid channel set for single-channel file. Using channel 1")

        signal = a

    trim_start = max(int(fs * trim_start) - 1, 0)
    if length > 0:
        length = trim_start + (fs * length)
        signal = signal[trim_start:length]
    else:
        signal = signal[trim_start:]

    if soundtrap:
        signal = soundtrap_conversion(signal, soundtrap)

    time = np.arange(0, signal.size / fs, 1/fs)
    if len(time) == len(signal) + 1:
        time = time[:-1]

    return fs, signal