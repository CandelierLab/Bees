'''

Avidemux crop parameters
6x: 1420 - 580  (500x500 pix)
4x: 3180 - 1500 (660x660 pix)
'''

import os
import numpy as np

import IP

os.system('clear')

# === Parameters ===========================================================

# xtype = 'x6'
xtype = 'x4'

movie_file = 'sNPF videos -  Motor activity/14062024/MVI_0015/MVI_0015_6.mp4'
# movie_file = 'sNPF videos -  Motor activity/20062024/C0001/C0001_4.mp4'

# ==========================================================================

P = IP.processor(xtype, movie_file)

# print(np.max(P.background))



# P.show(P.background)

# P.viewer()

# P.run(display=True)

if not os.path.exists(P.file['traj']):

  P.check_background()
  P.run(display=True)