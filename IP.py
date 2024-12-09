'''
Image processing tools

NB: RWU 430pix for 90mm 
'''

import os
import time
import warnings
import yaml
import numpy as np
import pandas as pd
import cv2 as cv
from alive_progress import alive_bar
import matplotlib.pyplot as plt

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

class handler:

  def __init__(self, stype, btype):

    # --- Data types

    self.btype = btype
    self.stype = stype
    self.type = f'{self.stype} {self.btype}'

    # --- Load csv

    self.csv_path = project.root + 'Data/' + f'2024 - sNPF - {self.type}.csv'
    self.df = pd.read_csv(self.csv_path)

class processor:

  def __init__(self, dtype, movie_code, dish, pix2mm=None, verbose=False):

    # --- Definitions

    self.type = dtype
    self.movie_code = movie_code
    self.dish = dish
    self.verbose = verbose

    self.th_area = None

    # --- Settings

    self.file = {}

    # Movie
    self.file['movie'] = {}
    self.file['movie']['filename'] = f'{self.movie_code}_{dish}.mp4'
    self.file['movie']['path'] = project.root + f'Data/{self.type}/{self.movie_code}/'+ self.file['movie']['filename']

    # Directory
    self.dir = project.root + f'Files/{self.type}/{self.movie_code}_{dish}/'
    if not os.path.exists(self.dir):
      os.makedirs(self.dir)

    # Parameters
    self.param = None
    self.file['parameters'] = self.dir + 'parameters.yml'
    if not os.path.exists(self.file['parameters']):

      # Frame size
      cap = cv.VideoCapture(self.file['movie']['path'])
  
      if cap.isOpened(): 
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

      # RWU conversion
      if pix2mm is None:
        self.get_pix2mm(cap)

      with open(self.file['parameters'], 'w') as pfile:

        pfile.write(f'ROI: [0, {width}, 0, {height}]\npix2mm: {pix2mm}\nbackground:\n  method: median\n  nFrames: 10')

    # Background image
    self.file['background'] = self.dir + 'background.npy'

    # CSV export
    self.file['traj'] = self.dir + 'trajectories.csv'

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
    return frame if self.param is None else frame[self.param['ROI'][2]:self.param['ROI'][3], self.param['ROI'][0]:self.param['ROI'][1]]

  def get_pix2mm(self, cap):

    # --- Initial frame    
    Img = self.get_frame(cap)
    cap.release()

    self.RWU_pts = []

    def click(event, x, y, flag, self):

      if event == cv.EVENT_LBUTTONDOWN: 
        '''
        Define point
        '''

        # Store points
        self.RWU_pts.append([x, y])

        if len(self.RWU_pts)==3:
      
          # Get points
          x1 = self.RWU_pts[0][0]
          x2 = self.RWU_pts[1][0]
          x3 = self.RWU_pts[2][0]
          y1 = self.RWU_pts[0][1]
          y2 = self.RWU_pts[1][1]
          y3 = self.RWU_pts[2][1]

          # Circle radius
          a = np.sqrt((x2-x3)**2 + (y2-y3)**2)
          b = np.sqrt((x1-x3)**2 + (y1-y3)**2)
          c = np.sqrt((x1-x2)**2 + (y1-y2)**2)
          s = (a+b+c)/2
          A = np.sqrt(s*(s-a)*(s-b)*(s-c))
          r = a*b*c/4/A

          # pix2mm coefficient (90mm Petri dishes)
          self.pix2mm = 45/r

          print(f'{self.movie_code}_{self.dish} | RWU conversion coefficient: {self.pix2mm}')

    # --- Display
    cv.imshow('frame', Img*2)
    cv.setMouseCallback('frame', click, self)

    cv.waitKey(0)
    cv.destroyAllWindows()

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

  def check_background(self):

    Src = self.background.astype(np.float32)
    Img = None

    # Radius 10mm
    r = 10/self.param['pix2mm']

    pt = None

    def click(event, x, y, *args):

      global Img

      Src = self.background.astype(np.float32)

      if event == cv.EVENT_LBUTTONDOWN: 
        '''
        Define region to suppress
        '''
        pt = [x, y]
      
        # Define mask
        X, Y = np.meshgrid(np.arange(Src.shape[1]), np.arange(Src.shape[0]))
        mask = np.uint8((X-pt[0])**2 + (Y-pt[1])**2 <= r**2)

        # New image
        Img = cv.inpaint(Src, mask, 2, cv.INPAINT_NS)

        # norm = cv.normalize(Img, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        norm = Img/np.mean(Img)/2

        cv.imshow('frame', norm)

      if event == cv.EVENT_MBUTTONDOWN:
        '''
        Save background
        '''

        np.save(self.file['background'], Img)
        print(f'{self.movie_code}_{self.dish} | New background saved.')

        self.background = Img
        Src = Img
        
    # Initial display
    # norm = cv.normalize(Src, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    norm = Src/np.mean(Src)/2
    cv.imshow('frame', norm)

    cv.setMouseCallback('frame', click)

    cv.waitKey(0)
    cv.destroyAllWindows()

  def show(self, Img):

    cv.imshow('frame', Img)
    cv.waitKey(0)
    cv.destroyAllWindows()

  def display_traj(self):

    # Background
    Bkg = self.background/np.mean(self.background)/2

    # Trajectory
    df = pd.read_csv(self.file['traj'])

    # plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 22})
    ax = plt.axes(projection='3d')

    # ax.imshow(Bkg)
    ax.plot3D(df.x, df.y, df.t, '-')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('t (s)')

    plt.show()

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

  # ========================================================================
  def process(self, Img, kind='Single'):

    Tmp = self.background - Img

    _, BW = cv.threshold(Tmp, 0.03, 1, cv.THRESH_BINARY)
    self.BW = BW
    
    match kind:

      case 'Single':

        # --- Find largest object      
        cnts, _ = cv.findContours(BW.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if not len(cnts): return [None]
        cnt = max(cnts, key=cv.contourArea)

        # Result
        BW = np.zeros(Img.shape, np.uint8)
        cv.drawContours(BW, [cnt], -1, 255, cv.FILLED)

        # --- Compute equivalent ellipse

        return [Ellipse(BW)]
      
      case 'Social':

        # --- Sort objects by area

        cnts, _ = cv.findContours(BW.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if not len(cnts): return [None, None]

        cnts = sorted(cnts, key=cv.contourArea, reverse=True)

        # Initialization of the area threshold
        if self.th_area is None and len(cnts)>1:
          self.th_area = (cv.contourArea(cnts[0]) + cv.contourArea(cnts[1]))/4

        # --- First ellipse

        if len(cnts)==0 or (self.th_area is not None and cv.contourArea(cnts[0])<self.th_area):
          return []
        else:
          BW = np.zeros(Img.shape, np.uint8)
          cv.drawContours(BW, [cnts[0]], -1, 255, cv.FILLED)
          E1 = Ellipse(BW)          

        # --- Second ellipse

        if len(cnts)<2 or cv.contourArea(cnts[1])<self.th_area:
          return [E1]

        else:
          BW = np.zeros(Img.shape, np.uint8)
          cv.drawContours(BW, [cnts[1]], -1, 255, cv.FILLED)
          E2 = Ellipse(BW)

        return [E1, E2]

  # ========================================================================
  def run(self, kind=None, display=False, save_csv=True, moviefile=None):
    '''
    Process the movie
    '''

    # --- Preparation ------------------------------------------------------

    if kind is None:
      if 'Single' in self.type: kind = 'Single'
      if 'Social' in self.type: kind = 'Social'

    # Input video 
    cap = cv.VideoCapture(self.file['movie']['path'])

    # Save result
    if save_csv:
      Data = []

    # Output video
    if moviefile is not None:
      vfile = cv.VideoWriter(moviefile, cv.VideoWriter_fourcc(*'MJPG'), fps=25, frameSize=(self.param['width'], self.param['height'])) 

    # --- Processing -------------------------------------------------------

    with alive_bar(self.param['T']-1) as bar:

      frame = 0
      t = 0
      Eref = None
      bmerge = None
      traces = [[]] if kind=='Single' else [[], []]

      bar.title(self.file['movie']['filename'][-14:-4])
      
      while cap.isOpened():

        Img = self.get_frame(cap)
        if Img is None: break
      
        # --- Processing ---------------------------------------------------

        E = self.process(Img, kind=kind)

        match kind:

          case 'Single':

            # --- Miss: use reference

            if E[0] is None:
              E = Eref
            else:
              Eref = E

            # --- Save

            if save_csv:
              Data.append([0, frame, t, E[0].x*self.param['pix2mm'], E[0].y*self.param['pix2mm'], E[0].theta, E[0].m00])

          case 'Social':

            if Eref is not None:

              match len(E):

                case 0:
                  E = Eref
                  bmerge = True if len(E)==1 else False

                case 1:

                  bmerge = True

                case 2:

                  bmerge = False

                  # --- Check inversions

                  if Eref is not None and len(Eref)>1:

                    # Squared distances to the references
                    d00 = (E[0].x-Eref[0].x)**2 + (E[0].y-Eref[0].y)**2
                    d01 = (E[0].x-Eref[1].x)**2 + (E[0].y-Eref[1].y)**2
                    d10 = (E[1].x-Eref[0].x)**2 + (E[1].y-Eref[0].y)**2
                    d11 = (E[1].x-Eref[1].x)**2 + (E[1].y-Eref[1].y)**2

                    if d00+d11 > d01+d10: E = [E[1], E[0]]

            # --- Reference update

            Eref = E

            # --- Save

            if save_csv:
              
              Data.append([0, frame, t,
                            E[0].x*self.param['pix2mm'], 
                            E[0].y*self.param['pix2mm'], 
                            E[0].theta, 
                            E[0].m00])
              
              if not bmerge and len(E)>1:
                Data.append([1, frame, t,
                             E[1].x*self.param['pix2mm'], 
                             E[1].y*self.param['pix2mm'], 
                             E[1].theta,
                             E[1].m00])

        # --- Display -------------------------------------------------------

        if display:

          # Images
          norm = cv.normalize(Img, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
          Res = cv.cvtColor(norm, cv.COLOR_GRAY2RGB)

          # Display BW images (instead of grayscale)
          # Res = cv.cvtColor(self.BW.astype(np.float32), cv.COLOR_GRAY2RGB)

          # --- Ellipses & traces

          # Reset traces
          if len(E)!=len(traces):
            traces = [[]] if bmerge else [[], []]

          for i in range(len(E)):

            if bmerge: color = (0,255,255)
            elif i: color = (255,255,0)
            else: color = (255,0,255)

            # Ellipse
            Res = cv.ellipse(Res, (int(E[i].x), int(E[i].y)), (int(E[i].l), int(E[i].w)), E[i].theta*180/np.pi, 0, 360, color=color, thickness=1)

            # Trace
            traces[i].append((int(E[i].x), int(E[i].y)))
            while len(traces[i])>50: traces[i].pop(0)
            
            Res = cv.polylines(Res, [np.array(traces[i]).reshape((-1, 1, 2))], False, color=color, thickness=1)

          # Display
          cv.imshow('frame', Res)

          # Save movie
          if moviefile:
            vfile.write(Res) 

          if cv.waitKey(1) == ord('q'):
            break

        # --- Update

        frame += 1
        t = frame/self.param['fps']
        bar()

    #  --- End -------------------------------------------------------------

    cap.release()
    if moviefile:
      vfile.release()

    cv.destroyAllWindows()

    # --- Save

    if frame>=self.param['T']-2 and save_csv:
      
      df = pd.DataFrame(Data, columns=['id', 'frame', 't', 'x', 'y', 'theta', 'area'])
      df.to_csv(self.file['traj'])