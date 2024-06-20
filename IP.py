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

class Ellipse:

  def __init__(self, Img):

    # --- Computing moments
    
    # Object's pixels coordinates
    J, I = np.where(Img)

    # Lambda function to compute the moments
    moment = lambda p, q: np.sum(np.power(I,p)*np.power(J,q))

    # --- Get image moments
    self.m00 = moment(0, 0)
    self.m10 = moment(1, 0)
    self.m01 = moment(0, 1)
    self.m11 = moment(1, 1)
    self.m02 = moment(0, 2)
    self.m20 = moment(2, 0)

    # --- Ellipse properties

    # Barycenter
    self.x = self.m10/self.m00
    self.y = self.m01/self.m00

    # Central moments (intermediary step)
    a = self.m20/self.m00 - self.x**2
    b = 2*(self.m11/self.m00 - self.x*self.y)
    c = self.m02/self.m00 - self.y**2

    # Orientation (radians)
    self.theta = 1/2*np.arctan(b/(a-c)) 
    self.theta += np.pi/2 if a<c else 0

    # Minor and major axis
    self.w = np.sqrt(8*(a+c-np.sqrt(b**2+(a-c)**2)))/2
    self.l = np.sqrt(8*(a+c+np.sqrt(b**2+(a-c)**2)))/2

    # % Ellipse focal points
    # d = sqrt(E.l^2-E.w^2);
    # E.x1 = E.x + d*cos(E.theta);
    # E.y1 = E.y + d*sin(E.theta);
    # E.x2 = E.x - d*cos(E.theta);
    # E.y2 = E.y - d*sin(E.theta);

    # % Ellipse direction
    # if direct
    #     tmp = [i-mean(i) j-mean(j)]*[cos(E.theta) -sin(E.theta) ; sin(E.theta) cos(E.theta)];
    #     if skewness(tmp(:,1))>0
            
    #         % Fix direction
    #         E.theta = mod(E.theta + pi, 2*pi);
    #         tmp = [E.x1 E.y1];
            
    #         % Swap F1 and F2
    #         E.x1 = E.x2;
    #         E.y1 = E.y2;
    #         E.x2 = tmp(1);
    #         E.y2 = tmp(2);
    #     end
    # end

class processor:

  def __init__(self, movie_file, verbose=True):

    # --- Definitions

    self.verbose = verbose

    # --- Settings

    self.dir = project.root + 'Files/' + movie_file[:-4] + os.sep
    if not os.path.exists(self.dir):
      os.makedirs(self.dir)

    self.file = {}

    # Parameters
    self.param = None
    self.file['parameters'] = self.dir + 'parameters.yml'
    if not os.path.exists(self.file['parameters']):
      with open(self.file['parameters'], 'w') as pfile:
        pfile.write('''ROI: [0, 500, 0, 500]
background: 
  method: median
  nFrames: 10''')

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

  def play(self, record_movie=False):
    '''
    Play the movie
    '''

    if record_movie:
      vfile = cv.VideoWriter('tracking_1.avi', cv.VideoWriter_fourcc(*'MJPG'), fps=25, frameSize=(self.param['width'], self.param['height'])) 

    cap = cv.VideoCapture(self.file['movie']['path'])

    trace = []

    while cap.isOpened():
    
      Img = self.get_frame(cap)
      if Img is None: break
    
      Tmp = self.background - Img

      _, BW = cv.threshold(Tmp, 0.03, 1, cv.THRESH_BINARY)
      
      # --- Find largest object      
      cnts, _ = cv.findContours(BW.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
      cnt = max(cnts, key=cv.contourArea)

      # Result
      BW = np.zeros(Img.shape, np.uint8)
      cv.drawContours(BW, [cnt], -1, 255, cv.FILLED)

      # --- Compute equivalent ellipse

      E = Ellipse(BW)

      # print(E.__dict__)

      # --- Display -------------------------------------------------------

      # Images
      norm = cv.normalize(Img, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
      Res = cv.cvtColor(norm, cv.COLOR_GRAY2RGB)

      # Ellipse
      Res = cv.ellipse(Res, (int(E.x), int(E.y)), (int(E.l), int(E.w)), E.theta*180/np.pi, 0, 360, color=(255,0,255), thickness=1)

      # --- Trace

      trace.append((int(E.x), int(E.y)))
      while len(trace)>50: trace.pop(0)
      
      pts = np.array(trace)
      Res = cv.polylines(Res, [pts.reshape((-1, 1, 2))], False, color=(0,0,255), thickness=1)

      # Display
      cv.imshow('frame', Res)

      # cv.waitKey(0)
      # break

      # Save
      if record_movie:
        vfile.write(Res) 

      if cv.waitKey(1) == ord('q'):
        break

    cap.release()
    if record_movie:
      vfile.release()

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
