import unittest
import numpy as np
import pandas as pd
from soundscapecode.filters import highpass
from soundscapecode.soundtrap import open_wav

class TestFilters(unittest.TestCase):
    def test_highpass(self):
        fs, sig = open_wav('data/7255.221112060000.wav', trim_start=3, soundtrap=7255)
        fltrd = highpass(sig, 200, fs)
        expected = np.squeeze(pd.read_csv("data/7255_broad.csv", header=None).values)
        self.assertTrue(np.allclose(fltrd, expected, 10e2))
        