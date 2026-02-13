'''
Manual IP
'''

import os
import cv2 as cv
import imreg_dft as ird

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

P = IP.processor(H.type, movie_code, dish)

print(P)

# --- Background --------------------------------

# P.check_background()

# --- Single frame ------------------------------

# Template
tmp = cv.imread('template.png')

cap = cv.VideoCapture(P.file['movie']['path'])

frame = P.get_frame(cap, 3000)

result = ird.similarity(frame, tmp, numiter=1)

print(result)


# Tmp = P.background - frame
# _, BW = cv.threshold(Tmp, 0.03, 1, cv.THRESH_BINARY)
    
# # Erode
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
# # kernel = np.ones((3, 3))
# res = cv.erode(Tmp, kernel, iterations=2)
# res[res<0] = 0

# norm = cv.normalize(frame, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
# cv.imshow("tmp", tmp)
# cv.waitKey(0)



# norm = cv.normalize(frame, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
# cv.imshow("test", norm)
# cv.waitKey(0)
# cv.destroyAllWindows()

# --- Run ---------------------------------------

# P.run(display=True, save_csv=False)
# P.run(display=True, save_csv=Falsemoviefile='/home/raphael/Science/Projects/Misc/Bees/Movies/test.mp4')