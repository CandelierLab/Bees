import os

import IP

os.system('clear')

# === Parameters ===========================================================

movie_file = 'Pilot_1/MVI_0005.MP4'

# ==========================================================================

P = IP.processor(movie_file)

# print(P)

# P.show(P.background)

# P.viewer()

P.play()