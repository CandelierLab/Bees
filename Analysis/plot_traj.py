import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

P.display_traj()


# plt.style.use('dark_background')
# ax = plt.axes(projection='3d')

# ax.plot3D(df.x, df.y, df.t, '-')

# plt.show()

