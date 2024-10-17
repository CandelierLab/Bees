'''
Manual IP
'''

import os

import IP

os.system('clear')

# === Parameters ===========================================================

stype = 'Single'      # 'Single' / 'Social'
btype = 'foragers'    # 'foragers' / 'nurses'

movie_code = 'MVI_0038'
dish = 1

# ==========================================================================

# Data handler

H = IP.handler(stype, btype)

P = IP.processor(H.type, movie_code, dish)

# P.get_pix2mm()
# print(P)


# P.check_background()

# print(np.max(P.background))



# P.show(P.background*2)

P.viewer()

# P.run(display=True, moviefile='/home/raphael/Science/Projects/Misc/Bees/Movies/test.mp4')

# if not os.path.exists(P.file['traj']):

#   P.check_background()
#   P.run(display=True)