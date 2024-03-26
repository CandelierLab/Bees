'''
Image processing tools
'''

import os
import time
import cv2 as cv
import project

class processor:

  def __init__(self, movie_file, verbose=True):

    # --- Definitions

    self.verbose = verbose

    # --- Directories and files

    self.file = {}
    self.dir = {}

    # Movie
    self.file['movie'] = {}
    self.file['movie']['filename'] = movie_file
    self.file['movie']['path'] = project.root + 'Data/' + movie_file

    self.dir = project.root + 'Files/' + movie_file[:-4] + os.sep

    # Background image
    self.file['background'] = self.dir + 'Background.png'

    # --- Images

    # Background
    self.define_background()
    
  def __str__(self):
    
    s = f'Files directory: {self.dir}'

    return s

  def define_background(self):

    # Check existence
    if os.path.exists(self.file['background']):

      print('Background existing')
      self.background = None

    else:

      if self.verbose:
        print('Computing background ... ', end='')
        tref = time.time()

      ROI_box = [450, 1450, 80, 1080]
      cap = cv.VideoCapture(self.file['movie']['path'])

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


      if self.verbose:
        print('{:.2f} sec'.format(time.time() - tref))
