'''

Avidemux crop parameters
6x: 1420 - 580  (500x500 pix)
4x: 3180 - 1500 (660x660 pix)
'''

import os

import IP

os.system('clear')

# === Parameters ===========================================================

movie_file = 'sNPF videos -  Motor activity/14062024/MVI_0015/MVI_0015_2.mp4'
# movie_file = 'sNPF videos -  Motor activity/20062024/C0001/C0001_1.mp4'

# ==========================================================================

P = IP.processor(movie_file)

# print(P)

# P.show(P.background)

# P.viewer()

# P.run(display=True)

if not os.path.exists(P.file['traj']):

  P.check_background()
  P.run(display=True)