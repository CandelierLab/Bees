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
dish = 1



# ==========================================================================

# Data handler

H = IP.handler(season, stype, btype)

P = IP.processor(H, movie_code, dish)

print(P)

# --- Background --------------------------------

# P.check_background()

P.run(display=True, save_csv=False)

sys.exit()

# --- Template ----------------------------------

# # # P.template = P.load_template()

# x0 = [580, 490]
# y0 = [650, 130]
# a0 = [1.7, 1.5]

# # # x0 = [550, 600]
# # # y0 = [140, 200]
# # # a0 = [-1.2, 1.5]


# --- Single frame ------------------------------

frame = P.get_frame(5000)

X, Y, A = P.process(frame, x0, y0, a0)




T0 = P.set_template(P.template, X[0], Y[0], A[0])
T1 = P.set_template(P.template, X[1], Y[1], A[1])
T = np.maximum(T0, T1)

res = P.background - frame
res[res<0] = 0
img = (res - np.min(res))/(np.max(res) - np.min(res))

P.show_fusion(T, img)
# sys.exit()