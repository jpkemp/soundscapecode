import unittest
import numpy as np
import pandas as pd
from soundscapecode import soundtrap

class TestSoundtrap(unittest.TestCase):
    def test_open_wav(self):
        fpath = "data/7255.221112060000.wav"
        fs, sound = soundtrap.open_wav(fpath, soundtrap=7255, trim_start=3)
        expected = np.squeeze(pd.read_csv("data/7255_calsound.csv", header=None).values)
        self.assertEqual(sound.shape[0], expected.shape[0])
        self.assertTrue(np.allclose(sound, expected, 5))
