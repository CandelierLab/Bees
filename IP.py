'''
Image processing tools
'''

import os
import time
import warnings
import yaml
import numpy as np
import cv2 as cv
import project

class processor:

  def __init__(self, movie_file, verbose=True):

    # --- Definitions

    self.verbose = verbose

    # --- Settings

    self.dir = project.root + 'Files/' + movie_file[:-4] + os.sep
    self.file = {}

    # Parameters
    self.param = None
    self.file['parameters'] = self.dir + 'parameters.yml'

    # Movie
    self.file['movie'] = {}
    self.file['movie']['filename'] = movie_file
    self.file['movie']['path'] = project.root + 'Data/' + movie_file

    # Background image
    self.file['background'] = self.dir + 'Background.npy'

    # --- Load associated data

    # Parameters
    self.load_parameters()

    # Background image
    self.define_background()
    
  def __str__(self):
    
    s = f'\n--- {self.__class__.__name__} ---\n'

    s += f'Files directory: {self.dir}\n'

    s += 'Parameters:\n'
    for k, v in self.param.items():
      s += f'- {k}: {v}\n'

    return s

  def load_parameters(self):
    '''
    Load the parameters from the movie and the associated YAML file.
    '''
    if not os.path.exists(self.file['parameters']):

      warnings.warn("Could not find the associated parameters file.")

    else:

      if self.verbose:
          print('Loading parameters ... ', end='')
          tref = time.time()

      # --- Read YAML file
          
      with open(self.file['parameters'], 'r') as stream:
        self.param = yaml.safe_load(stream)
        
      # --- Movie parameters
        
      # Width and eight
      self.param['width'] = self.param['ROI'][1]-self.param['ROI'][0]
      self.param['height'] = self.param['ROI'][3]-self.param['ROI'][2]

      # Duration and fps
      cap = cv.VideoCapture(self.file['movie']['path'])
      self.param['T'] = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
      self.param['fps'] = cap.get(cv.CAP_PROP_FPS)
      cap.release()

      if self.verbose:
        print('{:.2f} sec'.format(time.time() - tref))

  def get_frame(self, cap):
    '''
    Get a cropped frame in gray scale
    '''

    ret, frame = cap.read()
    if not ret: return None

    # Convert to grayscale
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)/255

    # Crop image
    Img = frame[self.param['ROI'][2]:self.param['ROI'][3], self.param['ROI'][0]:self.param['ROI'][1]]

    return Img

  def define_background(self):

    # Check existence
    if os.path.exists(self.file['background']):

      if self.verbose:
        print('Loading background ... ', end='')
        tref = time.time()

      self.background = np.load(self.file['background'])

      if self.verbose:
        print('{:.2f} sec'.format(time.time() - tref))

    else:

      if self.verbose:
        print('Computing background ... ', end='')
        tref = time.time()

      # Frame increment
      inc = int(self.param['T']/self.param['background']['nFrames'])

      if self.verbose:
        print(f'increment={inc} ... ', end='')
      
      # --- Stack ----------------------------------------------------------

      # Prepare stack        
      Stack = np.empty((self.param['height'], self.param['width'], self.param['background']['nFrames']))

      # --- Build stack
        
      cap = cv.VideoCapture(self.file['movie']['path'])
      t = 0

      while cap.isOpened():
      
        Img = self.get_frame(cap)
        if Img is None: break
        Stack[:,:,t] = Img
        t += 1

        cap.set(cv.CAP_PROP_POS_FRAMES, cap.get(cv.CAP_PROP_POS_FRAMES) + inc)
        
      cap.release()

      # --- Compute background

      match self.param['background']['method']:

        case 'median':

          self.background = np.median(Stack, axis=2)

      # --- Save background

      np.save(self.file['background'], self.background)

      if self.verbose:
        print('{:.2f} sec'.format(time.time() - tref))

  def show(self, Img):

    cv.imshow('frame', Img)
    cv.waitKey(0)
    cv.destroyAllWindows()

  def play(self):
    '''
    Play the movie
    '''

    cap = cv.VideoCapture(self.file['movie']['path'])

    while cap.isOpened():
    
      Img = self.get_frame(cap)
      if Img is None: break
    
      # Img = self.background - Img

      # _, BW = cv.threshold(Img, 0.03, 1, cv.THRESH_BINARY)

      cv.imshow('frame', Img)
      # cv.imshow('frame', BW)

      # cv.waitKey(0)
      # break

      if cv.waitKey(1) == ord('q'):
        break

    cap.release()
    cv.destroyAllWindows()

  def viewer(self):

    cap = cv.VideoCapture(self.file['movie']['path'])

    def onChange(t):
      cap.set(cv.CAP_PROP_POS_FRAMES, t)
      _, frame = cap.read()
      cv.imshow("mywindow", frame)

    cv.namedWindow('mywindow', cv.WINDOW_NORMAL)
    cv.createTrackbar('time', 'mywindow', 0, self.param['T']-1, onChange)

    onChange(0)
    while True:
      if cv.waitKey(1) == ord('q'):
        break
