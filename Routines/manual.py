'''
Manual IP
'''

import os, sys
import numpy as np
import cv2 as cv

import IP

os.system('clear')

# === Parameters ===========================================================

season = '2025 - Summer'
stype = 'Social'      # 'Single' / 'Social'
btype = 'foragers'    # 'foragers' / 'nurses'

movie_code = 'C0001'
dish = 3



# ==========================================================================

# Data handler

H = IP.handler(season, stype, btype)

P = IP.processor(H, movie_code, dish, verbose=False)

print(P)

# --- Background --------------------------------

# P.check_background()

# P.run(display=True, save_csv=False)
P.run(save_csv=False, moviefile=f'/home/raphael/Science/Projects/Misc/Bees/Movies/test/test_{movie_code}_{dish}.mp4')
