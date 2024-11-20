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
    calculates the temporal dissimilarity between two files

Classes
-------
SoundscapeCode: wrapper for calculating all metrics for all one-minute segments in a longer recording
'''
from ._soundscape_code import periodicity, max_spl, rms_spl, kurtosis, temporal_dissimilarity
from ._ssc import SoundscapeCode