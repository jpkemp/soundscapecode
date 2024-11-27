import unittest
import numpy as np
import pandas as pd
import soundscapecode as ssc
from soundscapecode import SoundscapeCode

class TestSoundscapeCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_data = "data"
        cls.test_file = f"{test_data}/7255.221112060000.wav"
        cls.fs, sound = ssc.soundtrap.open_wav(cls.test_file, soundtrap='data/7255.json', trim_start=3)
        one_min_interval = cls.fs * 60
        band_names = ("broad", "fish", "invertebrate")
        cls.freq_ranges = {"broad": (200, 24000), "fish": (200, 800), "invertebrate": (2000, 5000)}
        cls.sounds = {}
        cls.mean_freqs = {}
        for band in band_names:
            # don't want to replicate matlab filters so just load the data
            data = pd.read_csv(f"{test_data}/7255_{band}.csv", dtype=float, header=None).values
            sound_parts = [data[i:i+one_min_interval] for i in range(0, len(data), one_min_interval)]
            cls.sounds[band] = sound_parts
            assert len(cls.sounds[band]) == 5
            m_freq = np.squeeze(pd.read_csv(f"{test_data}/7255_{band}_mfreq.csv", header=None).values)
            cls.mean_freqs[band] = m_freq

        cls.validation = pd.read_csv(f"{test_data}/7255_SoundscapeCode.csv")
        cls.d_validation = pd.read_csv("data/dissimilarity.csv")
        cls.pxx_validation = pd.read_csv("data/7255_pxx.csv", header=None)
        cls.f, cls.t, cls.pxx = ssc.power_spectral_density(sound, cls.fs)

    def _compare_expected(self, band, sounds, metric, func, kwargs, rounding=7):
        expected = self.validation[f"Files_{metric}_{band}"]
        for i, data in enumerate(sounds[:4]):
            result = func(data, **kwargs)
            self.assertAlmostEqual(result, expected[i], rounding)

    def _compare_all_expected(self, metric, func, kwargs, rounding=7):
        for band, sounds in self.sounds.items():
            self._compare_expected(band, sounds, metric, func, kwargs, rounding)

    def test_periodicity(self):
        args = {'fs': self.fs}
        self._compare_all_expected("Acorr3", ssc.periodicity, args)

    def test_max_spl(self):
        args = {}
        self._compare_all_expected("Lppk", ssc.max_spl, args)

    def test_rms_spl(self):
        args = {'fs': self.fs}
        self._compare_all_expected("Lprms", ssc.rms_spl, args)

    def test_kurtosis(self):
        args = {}
        self._compare_all_expected("B", ssc.kurtosis, args, rounding=1)

    def test_temporal_dissimilarity(self):
        for band in self.sounds:
            expected = self.d_validation[f"Dt_{band}_Tobs"]
            for b in range(1, 4):
                a = b - 1
                data_a = self.sounds[band][a]
                data_b = self.sounds[band][b]
                dt = ssc.temporal_dissimilarity(data_a, data_b)
                ans = round(dt, 4)
                self.assertAlmostEqual(ans, expected[a])

    def test_psd(self):
        expected = self.pxx_validation
        test = self.pxx
        self.assertEqual(expected.shape, test.shape)
        self.assertEqual(self.t.shape[0], test.shape[1])
        self.assertEqual(self.f.shape[0], test.shape[0])
        self.assertTrue(np.allclose(expected, test, 5))

    def test_mfreq(self):
        for band, freq_range in self.freq_ranges.items():
            expected = self.mean_freqs[band]
            mfreq = ssc.meanfreq(self.pxx, self.f, freq_range)
            self.assertEqual(expected.shape, mfreq.shape)
            self.assertTrue(np.allclose(expected, mfreq, 5))

    def test_spectral_dissimilarity(self):
        for band, freq_range in self.freq_ranges.items():
            expected = self.d_validation[f"Df_{band}_Tobs"]
            mfreq = ssc.meanfreq(self.pxx, self.f, freq_range)
            for b in range(1, 4):
                a = b - 1
                freqs = []
                for idx in (a, b):
                    lower = idx * 120 # half second intervals, for one min period
                    upper = lower + 120
                    freq_part = mfreq[lower:upper]
                    freqs.append(freq_part)

                result = ssc.spectral_dissimilarity(*freqs)
                self.assertAlmostEqual(result, expected[a], places=2)

    def test_dissimilarity(self):
        for band, freq_range in self.freq_ranges.items():
            expected = self.validation[f"Files_D_{band}"]
            mfreq = ssc.meanfreq(self.pxx, self.f, freq_range)
            for b in range(1, 4):
                a = b - 1
                datas = []
                freqs = []
                for idx in (a, b):
                    lower = idx * 120 # half second intervals, for one min period
                    upper = lower + 120
                    data = self.sounds[band][idx]
                    freq_part = mfreq[lower:upper]
                    datas.append(data)
                    freqs.append(freq_part)

                result = ssc.dissimilarity_index(*datas, *freqs)
                self.assertAlmostEqual(result, expected[a], places=2)

    def test_ssc(self):
        for band, sounds in self.sounds.items():
            full_sound = np.concatenate(sounds)
            freq_range = self.freq_ranges[band]
            soundscape = SoundscapeCode(full_sound, self.fs, freq_range)
            self.assertEqual(soundscape.fs, self.fs)
            n = 5
            self.assertEqual(len(soundscape.sounds), n)
            self.assertEqual(len(soundscape.kurtosis), n)
            self.assertEqual(len(soundscape.periodicity), n)
            self.assertEqual(len(soundscape.Lppk), n)
            self.assertEqual(len(soundscape.Lprms), n)
            self.assertEqual(len(soundscape.temporal_dissimilarities), n-2)

            for metric in ["Acorr3", "Lppk", "Lprms", "B"]:
                expecteds = self.validation[f"Files_{metric}_{band}"]
                for i, expected in enumerate(expecteds):
                    expected = round(expected,1)
                    test = round(soundscape[metric][i],1)
                    self.assertEqual(test, expected)

            expecteds = self.d_validation[f"Dt_{band}_Tobs"]
            for i, expected in enumerate(expecteds[:3]):
                test = round(soundscape['dt'][i],4)
                self.assertEqual(test, expected)

            expecteds = self.d_validation[f"Df_{band}_Tobs"]
            for i, expected in enumerate(expecteds[:3]):
                test = round(soundscape['df'][i],4)
                self.assertAlmostEqual(test, expected, 2)

            expecteds = self.validation[f"Files_D_{band}"]
            for i, expected in enumerate(expecteds[:3]):
                test = round(soundscape['d'][i],4)
                self.assertAlmostEqual(test, expected, 2)
