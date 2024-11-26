'''Python implemention of the soundscapecode, for analysing ecoacoustic soundscapes.

See the `paper`_.
This implementation is ported from a Matlab `example`_.

.. _paper: https://pubs.aip.org/asa/jasa/article/149/4_Supplement/A72/651895/Introduction-and-application-of-a-proposed-method
.. _example: https://www.mathworks.com/matlabcentral/fileexchange/172434-sscmetrics-a-matlab-tool-to-compute-the-soundscape-code

Functions
----------
max_spl:
    calculates the maximum sound pressure level for a sound
rms_spl:
    calculates the root mean square sound pressure level for a sound
periodicity:
    calculates the periodicity for a sound
kurtosis:
    calculates the kurtosis for a sound
temporal_dissimilarity:
    calculates the temporal dissimilarity between two sound segments
stft_psd:
    calculates the power spectral density for a recording
meanfreq:
    gets the mean frequency per 0.1s segment in a recording, for a given frequency range
spectral_dissimilarity:
    calculates the spectral dissimilarity between two sound segments
dissimilarity index:
    calculates the dissimilarity index between two sound segments


Classes
-------
SoundscapeCode: wrapper for calculating all metrics for all one-minute segments in a longer recording
'''
from ._soundscape_code import periodicity, max_spl, rms_spl, kurtosis, temporal_dissimilarity, \
    power_spectral_density, meanfreq, spectral_dissimilarity, dissimilarity_index
from ._ssc import SoundscapeCode
from . import soundtrap