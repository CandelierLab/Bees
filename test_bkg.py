import numpy as np
import cv2 as cv

# === Parameters ===========================================================

mname = '/home/raphael/Science/Projects/Misc/Bees/Data/Videos - sNPF Pilot - 2024/MVI_0005.MP4'

# Box ROI (for initial crop)
ROI_box = [450, 1450, 80, 1080]     # x0, x1, y0, y1

# ==========================================================================

backSub = cv.createBackgroundSubtractorKNN()

cap = cv.VideoCapture(mname)

if not cap.isOpened():
 print('Unable to open.')
 exit(0)
  
while True:
 
 ret, frame = cap.read()

 if frame is None:
  break
 
 fgMask = backSub.apply(frame)
 
 cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
 cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
  
 cv.imshow('Frame', frame)
 cv.imshow('FG Mask', fgMask)
 
 
 keyboard = cv.waitKey(30)
 if keyboard == 'q' or keyboard == 27:
  break  