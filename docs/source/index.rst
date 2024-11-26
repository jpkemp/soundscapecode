.. soundscapecode documentation master file, created by
   sphinx-quickstart on Wed Nov 20 12:09:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**************
soundscapecode
**************

.. currentmodule:: soundscapecode

Python implemention of the soundscapecode, for analysing ecoacoustic soundscapes.

See the `paper`_.
This implementation is ported from a Matlab `example`_.

.. _paper: https://pubs.aip.org/asa/jasa/article/149/4_Supplement/A72/651895/Introduction-and-application-of-a-proposed-method
.. _example: https://www.mathworks.com/matlabcentral/fileexchange/172434-sscmetrics-a-matlab-tool-to-compute-the-soundscape-code

Additionally, the soundscapecode.soundtrap sub-module is provided for working with data from soundtrap hydrophones.

.. include:: usage.rst

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   SoundscapeCode
   soundtrap.CalibrationData

.. autosummary::
   :toctree: _autosummary

   max_spl
   rms_spl
   periodicity
   kurtosis
   temporal_dissimilarity
   spectral_dissimilarity
   dissimilarity_index
   soundtrap.soundtrap_conversion
   soundtrap.get_soundtrap_calibration
   soundtrap.read_calibration
   soundtrap.open_wav
