# SoundScapeCode

Python implementation of the soundscape code [[1]](#1), ported from a Matlab example [[2]](#2).

## Installation
```
pip install soundscapecode
```

## Usage
Values can be calculated individually:
```
import soundscapecode as ssc

fs = 48000
mock_sound = np.random.rand(fs*60, 1)
periodicity = ssc.periodicity(mock_sound, fs)
```

Or to calculate all values for one recording:
```
from soundscapecode import SoundscapeCode

fs = 48000
n_mins = 3
mock_sound = np.random.rand(fs*n_mins*60, 1)
soundscape = SoundscapeCode(mock_sound, fs)
for dt in soundscape["dt"]:
    print(dt)
```

## Documentation
Check the documentation at <a href=https://soundscapecode.readthedocs.io/en/latest>readthedocs</a>.

## Contact
Please contact me with any questions
<a href=https://au.linkedin.com/in/james-kemp-11874a93><img src=https://blog-assets.hootsuite.com/wp-content/uploads/2018/09/In-2C-54px-R.png
    width = 18 height = 15 /></a>
<a href=https://www.researchgate.net/profile/James_Kemp6><img src=https://www.researchgate.net/apple-touch-icon-180x180.png
    width=15 height=15 /></a>

## References
<a id="1">[1]</a>
Dylan Wilford, Jennifer Miksis-Olds, Bruce Martin, Kim Lowell; Introduction and application of a proposed method for quantitative soundscape analysis: The soundscape code. J. Acoust. Soc. Am. 1 April 2021; 149 (4_Supplement): A72. https://doi.org/10.1121/10.0004555
</br>
<a id="2">[2]</a>
https://www.mathworks.com/matlabcentral/fileexchange/172434-sscmetrics-a-matlab-tool-to-compute-the-soundscape-code