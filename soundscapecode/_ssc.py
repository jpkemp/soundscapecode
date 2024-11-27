import numpy as np
from soundscapecode import _soundscape_code as ssc

class SoundscapeCode:
    '''Wrapper for segmenting and calculating soundscape code metrics for a sound.

    Attributes
    ----------
    sound: np.ndarray
        the given sound recording from which the metrics were calculated
    fs: int
        sampling frequency for the sound
    freq_range: tuple
        frequency range for spectral dissimilarity, if given
    Lppk: list
        Peak SPL values at one-minute intervals
    Lprms: list
        Root mean squared SPL values at one-minute intervals
    kurtosis: list
        kurtosis values at one-minute intervals
    periodicity: list
        periodicity values at one-minute intervals
    temporal_dissimilarities: list
        temporal dissimilarities between consecutive one-minute intervals
    spectral_dissimilarities: list
        spectral dissimilarities between consecutive one-minute intervals
    spectral_dissimilarities: list
        spectral dissimilarities between consecutive one-minute intervals
    dissimilarities: list
        dissimilarity index between consecutive one-minute intervals

    Examples
    -----
    >>> import numpy as np
    >>> from soundscapecode import SoundscapeCode
    >>> fs = 48000
    >>> n_mins = 3
    >>> sound = np.random.rand(fs*n_mins*60,1)
    >>> soundscape = SoundscapeCode(sound, fs)
    >>> for pk_spl in soundscape["max_spl"]:
    ...    print(pk_spl)
    -2.786002960850315e-06
    -6.53336810900092e-06
    -7.38333472594301e-06
    '''

    def __init__(self, sound:np.ndarray, fs:int, freq_range=None):
        '''
        Parameters
        ----------
        sound: np.ndarray
            the sound to analyse. The sound will be segmented into one-minute blocks.
        fs: int
            sampling frequency for the sound
        '''
        one_min_interval = fs * 60
        self.sound:np.ndarray = sound
        self.fs:int = fs
        self.sounds:list[np.ndarray] = []
        self.kurtosis:list[np.float] = []
        self.periodicity:list[np.float] = []
        self.Lppk:list[np.float] = []
        self.Lprms:list[np.float] = []
        self.dissimilarities:list[np.float] = []
        self.spectral_dissimilarities:list[np.float] = []
        self.temporal_dissimilarities:list[np.float] = []
        self.freq_range = freq_range
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
        if lower in ["df", "spectral", "spectral_dissimilarity", "dissimilarity_spectral"]:
            return self.spectral_dissimilarities
        if lower in ["d", "dissimilarity", "dissimilarity_index"]:
            return self.dissimilarities

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

        f, t, pxx = ssc.power_spectral_density(self.sound, self.fs)
        meanfreqs = ssc.meanfreq(pxx, f, self.freq_range)
        one_min = 120
        remainder = t.shape[0] % one_min
        for segment_start in range(0, t.shape[0] - remainder - one_min, one_min):
            a = meanfreqs[segment_start:segment_start+one_min]
            b = meanfreqs[segment_start+one_min:segment_start+2*one_min]
            spectral_dissimilarity = ssc.spectral_dissimilarity(a, b)
            self.spectral_dissimilarities.append(spectral_dissimilarity)

        max_idx = min(len(self.temporal_dissimilarities), len(self.spectral_dissimilarities))
        self.dissimilarities = list(np.array(self.temporal_dissimilarities[:max_idx]) * 
                                    np.array(self.spectral_dissimilarities[:max_idx]))
