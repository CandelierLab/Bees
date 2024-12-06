'''
IP routine
'''

import os
import numpy as np
import pandas as pd
import cv2 as cv
from alive_progress import alive_bar
import matplotlib.pyplot as plt

import IP

os.system('clear')

# === Parameters ===========================================================

stype = 'Single'      # 'Single' / 'Social'
btype = 'foragers'    # 'foragers' / 'nurses'

bsize = 250

# ==========================================================================

# Data handler

H = IP.handler(stype, btype)

row = H.df.iloc[103]

print(row)
movie_code = row['video code']
dish = row['petri dish place']
pix2mm = row['pix2mm']

P = IP.processor(H.type, movie_code, dish, pix2mm=pix2mm)

# Trajectory
df = pd.read_csv(P.file['traj'])

# print(df.y[0])

# === Image processing =====================================================

plt.style.use('dark_background')
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(1,1, figsize=(20,20))

theta = np.unwrap(2*df.theta)/2

# # # ax.plot(theta, '.-')

# # # plt.show()

# Input video 
cap = cv.VideoCapture(P.file['movie']['path'])

with alive_bar(P.param['T']-1) as bar:

  frame = 0
  t = 0
  id = 0
  Eref = None

  bar.title(P.file['movie']['filename'][-14:-4])
  
  while cap.isOpened():

    Img = P.get_frame(cap)
    if Img is None: break
  
    # --- Processing ---------------------------------------------------

    dx = df.x[frame]/P.param['pix2mm']
    dy = df.y[frame]/P.param['pix2mm']
    
    # M = cv.getRotationMatrix2D((dx, dy), df.theta[frame]*180/np.pi+90, 1)
    M = cv.getRotationMatrix2D((dx, dy), theta[frame]*180/np.pi+90, 1)
    M[0,2] += bsize/2 - dx
    M[1,2] += bsize/2 - dy

    Img = P.background - Img
    Res = cv.warpAffine(Img, M, [bsize]*2)
    
    # --- Display -------------------------------------------------------
    
    # Determine head
    # if np.argmax(np.sum(Res, axis=1))>bsize/2:
    Res = np.flip(Res, axis=0)

    # if frame==0:
    #   # ax.imshow(Res, cmap='gray', vmax=0.1)
    #   ax.plot(np.sum(Res, axis=1), '.-')
    #   plt.show()
    #   break

    # Upscale
    Res = cv.resize(Res, (500,500), interpolation=cv.INTER_LINEAR)*4

    # Display
    cv.imshow('frame', 1-Res)

    if cv.waitKey(1) == ord('q'):
      break

    # --- Update

    frame += 1
    t = frame/P.param['fps']
    bar()

#  --- End -------------------------------------------------------------

cap.release()
cv.destroyAllWindows()