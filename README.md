# FacialMotionCapture

See details here:
https://youtu.be/O7nNO3FLkLU

I WILL NOT EXPLAIN HOW TO INSTALL MODULES TO BLENDER'S PYTHON. LOOK UP A TUTORIAL!
A little hint though for PYAudio, don't use pip directly!!! Install it from a .whl file from [this](https://www.lfd.uci.edu/~gohlke/pythonlibs/) website and put the path of the downloaded file in place of the module name (in this case, "pyaudio")

This is an upgrade of the addon:
- no longer only does vincent. Just any model you want
- warns that you don't have CV2 or my own addition PYAudio
- added ui for choosing bones
- added audio form detection for mouth movement
- added model training data download link and training data location chooser
- works for Blender 3.0.0
