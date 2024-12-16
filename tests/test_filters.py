import unittest
import numpy as np
import pandas as pd
import soundscapecode.filters as flt
from soundscapecode.soundtrap import open_wav

class TestFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs, cls.sig = open_wav('data/7255.221112060000.wav', trim_start=3, soundtrap=7255)
        return super().setUpClass()

    def test_taps(self):
        for name, band in [("broad", (200,)),
                            ("fish", (200, 800)),
                            ("invertebrate", (2000, 5000))]:
            fltr = "highpass" if len(band) == 1 else "bandpass"
            taps = flt._get_valid_kaiser_filter_window(band, 48000, 60, filter_type=fltr)
            expected = np.squeeze(pd.read_csv(f"data/taps_{name}.csv", header=None))
            self.assertTrue(np.allclose(taps, expected))

    def test_highpass(self):
        fltrd = flt.highpass(self.sig, 200, self.fs)
        expected = np.squeeze(pd.read_csv("data/7255_broad.csv", header=None).values)
        self.assertTrue(np.allclose(fltrd, expected, atol=10e2))

    def test_bandpass(self):
        for fl, freqs in [("data/7255_fish.csv", (200, 800)), 
                          ("data/7255_invertebrate.csv", (2000, 5000))]:
            expected = np.squeeze(pd.read_csv(fl, header=None).values)
            fltrd = flt.bandpass(self.sig, freqs, self.fs)
            self.assertTrue(np.allclose(fltrd, expected, atol=10e2))
