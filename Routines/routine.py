import os

import IP

os.system('clear')

# === Parameters ===========================================================

movie_file = 'sNPF videos -  Motor activity/14062024/MVI_0012/MVI_0012_3.mp4'

# ==========================================================================

P = IP.processor(movie_file)

# print(P)

# P.show(P.background)

# P.viewer()

if not os.path.exists(P.file['traj']):

  P.check_background()
  P.run(display=True)