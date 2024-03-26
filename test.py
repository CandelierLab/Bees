import numpy as np
import cv2 as cv

# === Parameters ===========================================================

mname = '/home/raphael/Science/Projects/Misc/Bees/Data/Pilot_1/MVI_0005.MP4'

# Box ROI (for initial crop)
ROI_box = [450, 1450, 80, 1080]     # x0, x1, y0, y1

# ==========================================================================

cap = cv.VideoCapture(mname)

while cap.isOpened():
 
  ret, frame = cap.read()
 
  # if frame is read correctly ret is True
  if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break
 
  # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
  Img = frame[ROI_box[2]:ROI_box[3], ROI_box[0]:ROI_box[1]]

  cv.imshow('frame', Img)

  # cv.waitKey(0)
  # break

  if cv.waitKey(1) == ord('q'):
    break
  
cap.release()
cv.destroyAllWindows()