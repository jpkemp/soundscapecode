Installation and Usage
----------------------
``pip install soundscapecode``

Values can be calculated individually:

.. code:: python

    import soundscapecode as ssc

    fs = 48000
    mock_sound = np.random.rand(fs*60, 1)
    periodicity = ssc.periodicity(mock_sound, fs)


Or to calculate all values for one recording:

.. code:: python

    from soundscapecode import SoundscapeCode

    fs = 48000
    n_mins = 3
    mock_sound = np.random.rand(fs*n_mins*60, 1)
    soundscape = SoundscapeCode(mock_sound, fs)
    for dt in soundscape["dt"]:
        print(dt)
