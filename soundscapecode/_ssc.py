import numpy as np
from soundscapecode import _soundscape_code as ssc

class SoundscapeCode:
    '''Wrapper for segmenting and calculating soundscape code metrics for a sound.

    Parameters
    ----------
    sound: np.ndarray
        the sound to analyse. The sound will be segmented into one-minute blocks.
    fs: int
        sampling frequency for the sound

    Examples
    -----
    >>>import numpy as np
    >>>from soundscapecode import SoundscapeCode
    >>>fs = 48000
    >>>n_mins = 3
    >>>sound = np.random.rand(fs*n_mins*60,1)
    >>>soundscape = SoundscapeCode(sound, fs)
    >>>for pk_spl in soundscape["max_spl"]:
    ...    print(pk_spl)
    -2.786002960850315e-06
    -6.53336810900092e-06
    -7.38333472594301e-06
    '''

    def __init__(self, sound:np.ndarray, fs:int):
        one_min_interval = fs * 60
        sound:np.ndarray = sound
        self.fs:int = fs
        self.sounds:list[np.ndarray] = []
        self.kurtosis:list[np.float] = []
        self.periodicity:list[np.float] = []
        self.Lppk:list[np.float] = []
        self.Lprms:list[np.float] = []
        self.temporal_dissimilarities:list[np.float] = []
        for i in range(0, len(sound), one_min_interval):
            self.sounds.append(sound[i:i+one_min_interval])

        self._calculate_metrics()

    def __getitem__(self, item):
        lower = item.lower()
        if lower in ["kurtosis", "impulsivity", "b"]:
            return self.kurtosis
        if lower in ["periodicity", "acorr3"]:
            return self.periodicity
        if lower in ["rms", "lprms", "rms_spl", "spl_rms"]:
            return self.Lprms
        if lower in ["max", "lppk", "max_spl", "spl_max", "pk_spl"]:
            return self.Lppk
        if lower in ["dt", "temporal", "temporal_dissimilarity", "dissimilarity_temporal"]:
            return self.temporal_dissimilarities
        
        return NotImplemented

    def _calculate_metrics(self):
        '''Calculate all metrics

        :meta private:
        '''
        for data in self.sounds:
            self.Lppk.append(ssc.max_spl(data))
            self.Lprms.append(ssc.rms_spl(data, self.fs))
            self.kurtosis.append(ssc.kurtosis(data))
            self.periodicity.append(ssc.periodicity(data, self.fs))

        for i, data in enumerate(self.sounds[:-1]):
            a = self.sounds[i]
            b = self.sounds[i+1]
            if a.size != b.size:
                continue

            dis = ssc.temporal_dissimilarity(a, b)
            self.temporal_dissimilarities.append(dis)
