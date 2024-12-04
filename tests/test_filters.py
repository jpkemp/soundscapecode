import unittest
import numpy as np
import pandas as pd
from soundscapecode.filters import highpass, bandpass
from soundscapecode.soundtrap import open_wav

class TestFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs, cls.sig = open_wav('data/7255.221112060000.wav', trim_start=3, soundtrap=7255)
        return super().setUpClass()

    def test_highpass(self):
        fltrd = highpass(self.sig, 200, self.fs)
        expected = np.squeeze(pd.read_csv("data/7255_broad.csv", header=None).values)
        self.assertTrue(np.allclose(fltrd, expected, 10e2))

    def test_bandpass(self):
        for fl, freqs in [("data/7255_fish.csv", (200, 800)), 
                          ("data/7255_invertebrate.csv", (2000, 5000))]:
            expected = np.squeeze(pd.read_csv(fl, header=None).values)
            fltrd = bandpass(self.sig, *freqs, self.fs)
            self.assertTrue(np.allclose(fltrd, expected, 10e2))
