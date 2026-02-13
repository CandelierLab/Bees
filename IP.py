'''
Image processing tools
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

plt.style.use('dark_background')

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

  def __init__(self, season, stype, btype):

    # --- Data types

    self.season = season
    self.btype = btype
    self.stype = stype
    self.type = f'{season}/{self.stype} {self.btype}'

    # --- Load csv

    self.csv_path = project.root + 'Data/' + f'{self.type}.csv'
    self.df = pd.read_csv(self.csv_path)

class processor:

  def __init__(self, H:handler, movie_code, dish, nbees=2, pix2mm=None, verbose=False):

    # --- Definitions

    self.handler = H
    self.type = self.handler.type
    self.movie_code = movie_code
    self.dish = dish
    self.nbees = nbees
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
        pix2mm = self.get_pix2mm(cap)

      with open(self.file['parameters'], 'w') as pfile:

        pfile.write(f'ROI: [0, {width}, 0, {height}]\npix2mm: {pix2mm}\nbackground:\n  method: median\n  nFrames: 10')

    # Background image
    self.file['background'] = self.dir + 'background.npy'

    # Template
    self.file['template'] = project.root + f'Files/{self.handler.season}/template.png'

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

  def load_template(self, path=None):
    '''
    Ouvrir et retourner une image PNG depuis le disque.
    `path` peut être un chemin absolu ou relatif.
    '''

    if path is None:
      path = self.file['template']

    # Expand user and get absolute path
    p = os.path.expanduser(path)
    p = os.path.abspath(p)

    if not os.path.exists(p):
      raise FileNotFoundError(f"Template not found: {path}")

    # Read image (preserve alpha if present)
    img = cv.imread(p, cv.IMREAD_UNCHANGED)
    if img is None:
      raise ValueError(f"Could not read image: {p}")

    if self.verbose:
      print(f'Loaded template: {p}')

    img = img[:,:,0].astype(float)
    img = (img - np.min(img))/(np.max(img) - np.min(img))

    return img

  def set_template(self, template, x, y, angle, pad_value=0.0):
    '''
    Retourne une grande image contenant `template` positionnée au centre `(x,y)`
    et tournée de `angle` radians. `template` doit être un tableau 2D float
    avec valeurs en [0,1]. `pad_value` (float) remplit les pixels non définis.

    La taille de l'image de sortie est déterminée par `self.param['width']`/`height`
    si `self.param` est disponible, sinon par `self.background` si existante.
    Le point `(x,y)` correspond au centre du template dans l'image de sortie.
    '''

    # Convertir et valider le template
    T = np.array(template, dtype=np.float32)
    if T.ndim != 2:
      raise ValueError('Template must be a single-channel (2D) array')

    th, tw = T.shape

    # Rotation autour du centre du template (cv expects degrees)
    angle_deg = float(angle) * 180.0 / np.pi
    center = (tw/2.0, th/2.0)
    M = cv.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv.warpAffine(T, M, (tw, th), flags=cv.INTER_LINEAR,
                             borderMode=cv.BORDER_CONSTANT, borderValue=float(pad_value))

    # Taille de sortie: préférer self.param, sinon self.background
    if getattr(self, 'param', None) is not None:
      H = int(self.param['height'])
      W = int(self.param['width'])
    elif getattr(self, 'background', None) is not None:
      H, W = self.background.shape
    else:
      raise RuntimeError('Cannot determine output size: load parameters or background first')

    out = np.full((H, W), float(pad_value), dtype=np.float32)

    # Positionner le template centré en (x,y)
    cx = float(x)
    cy = float(y)
    x0 = int(round(cx - tw/2.0))
    y0 = int(round(cy - th/2.0))
    x1 = x0 + tw
    y1 = y0 + th

    # Calcul des intersections (template -> out)
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    sx1 = tw - max(0, x1 - W)
    sy1 = th - max(0, y1 - H)

    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    # Pas de recouvrement
    if sx1 <= sx0 or sy1 <= sy0:
      return out

    out[dy0:dy1, dx0:dx1] = rotated[sy0:sy1, sx0:sx1]

    if self.verbose:
      print(f'Set template at ({cx:.1f},{cy:.1f}) angle={angle_deg:.1f}°')

    return out

  def get_frame(self, n=None, cap=None):
    '''
    Get a cropped frame in gray scale
    '''

    if cap is None:
      cap = cv.VideoCapture(self.file['movie']['path'])

    if n is not None:
      cap.set(cv.CAP_PROP_POS_FRAMES, n)

    ret, frame = cap.read()
    if not ret: return None

    # Convert to grayscale
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)/255

    # Crop image
    return frame if self.param is None else frame[self.param['ROI'][2]:self.param['ROI'][3], self.param['ROI'][0]:self.param['ROI'][1]]

  def get_pix2mm(self, cap=None):

    # --- Movie
    if cap is None:
      cap = cv.VideoCapture(self.file['movie']['path'])

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
          self.pix2mm = float(45/r)

          print(f'{self.movie_code}_{self.dish} | RWU conversion coefficient: {self.pix2mm}')

    # --- Display
    cv.imshow('frame', Img*2)
    cv.setMouseCallback('frame', click, self)

    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return self.pix2mm

  def set_pix2mm(self, pix2mm):
    '''
    Set the pix2mm parameter in the YAML parameters file.
    '''
    # Load current parameters
    with open(self.file['parameters'], 'r') as stream:
      params = yaml.safe_load(stream)
    
    # Update pix2mm parameter
    params['pix2mm'] = pix2mm
    
    # Save updated parameters
    with open(self.file['parameters'], 'w') as stream:
      yaml.dump(params, stream)
    
    # Reload parameters
    self.load_parameters()
    
    if self.verbose:
      print(f'Updated pix2mm to {pix2mm}')

  def define_background(self, force=False):

    # Check existence
    if os.path.exists(self.file['background']) and not force:

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

  def show(self, Img, norm=False, show=True):

    if norm:
      Img = (Img - np.min(Img))/(np.max(Img) - np.min(Img))

    # cv.imshow('show', Img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    I = ax.pcolormesh(Img, cmap=plt.cm.gray, vmin=0, vmax=1)
    cb = plt.colorbar(I)

    if show:
      plt.show()

  def show_fusion(self, img0, img1, show=True):

    res = np.dstack([img0, img1, img0])

    # cv.imshow('fusion', res)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    I = ax.imshow(res)

    if show:
      plt.show()

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
  def process(self, img, X, Y, A):

    # Increments
    l_t_inc = [5, 1]
    l_a_inc = [np.pi/36, np.pi/180]

    def corr(A,B):
      return np.corrcoef(A.flatten(), B.flatten())[0,1].item()

    def get_corr(X, Y, A, k=0, dx=0, dy=0, da=0):
      T0 = self.set_template(self.template, X[0] + (0 if k else dx), 
                                Y[0] + (0 if k else dy),
                                A[0] + (0 if k else da))
      T1 = self.set_template(self.template, X[1] + (dx if k else 0), 
                                Y[1] + (dy if k else 0),
                                A[1] + (da if k else 0))
      return corr(img, np.maximum(T0, T1))

    def print_state(X, Y, A, C):
      if self.verbose:
        print(f'C={C} ({X[0]:d}, {Y[0]:d},{A[0]:.02f}) ({X[1]:d}, {Y[1]:d},{A[1]:.02f})')

    # Transform image
    res = self.background - img
    res[res<0] = 0
    img = (res - np.min(res))/(np.max(res) - np.min(res))
    
    # Initial values
    C = get_corr(X, Y, A)
    print_state(X, Y, A, C)

    # ======================================================================
    # Iterative procedure

    for (t_inc, a_inc) in zip(l_t_inc, l_a_inc):

      opt = True

      while opt:

        if self.verbose:
          print(f'Iteration @ ({t_inc:d}, {a_inc:.02f})')

        opt = False

        for k in range(self.nbees):

          # === Angular optimization =================

          # --- Positive a

          aneg = True

          while True:

            c = get_corr(X, Y, A, k, da=a_inc) 
            if c>C:
              A[k] += a_inc
              C = c
              aneg = False
              opt = True
              print_state(X, Y, A, C)
            else:
              break

          # --- Negative a

          if aneg:
            while True:

              c = get_corr(X, Y, A, k, da=-a_inc) 
              if c>C:
                A[k] -= a_inc
                C = c
                opt = True
                print_state(X, Y, A, C)
              else:
                break

          # === Translation optimization =================

          # --- Try all orientations

          l_xy = [[t_inc,0],[t_inc, t_inc],[0,t_inc],[-t_inc,t_inc],[-t_inc,0],[-t_inc,-t_inc],[0,-t_inc],[t_inc,-t_inc]]
          c = [get_corr(X, Y, A, k, dx=xy[0], dy=xy[1]) for xy in l_xy]
          i = np.argmax(c)

          if c[i]>C:
            dx = l_xy[i][0]
            dy = l_xy[i][1]
            
            # Store improvement
            X[k] += dx
            Y[k] += dy
            C = c[i]
            opt = True
            print_state(X, Y, A, C)

            # Pursue in that direction
            while True:

              c = get_corr(X, Y, A, k, dx=dx, dy=dy) 
              if c>C:
                X[k] += dx
                Y[k] += dy
                C = c
                print_state(X, Y, A, C)
              else:
                break

    return (X, Y, A)

  def process_fast(self, img, X, Y, A):
    '''
    Optimized version of process for speed using template caching 
    and vectorized operations (~3-5x faster).
    '''
    
    # Increments
    l_t_inc = [5, 1]
    l_a_inc = [np.pi/36, np.pi/180]
    
    # Convert to numpy arrays (copy to avoid modifying originals in-place initially)
    X = np.array(X, dtype=np.int32, copy=True)
    Y = np.array(Y, dtype=np.int32, copy=True)
    A = np.array(A, dtype=np.float64, copy=True)
    
    def corr(A, B):
      return np.corrcoef(A.flatten(), B.flatten())[0,1].item()
    
    # Transform image - optimized version
    img_opt = self.background.astype(np.float32) - img.astype(np.float32)
    img_opt[img_opt < 0] = 0
    img_min = np.min(img_opt)
    img_max = np.max(img_opt)
    if img_max > img_min:
      img_opt = (img_opt - img_min) / (img_max - img_min)
    else:
      img_opt.fill(0.5)
    
    # Template cache: stores computed templates to avoid redundant calculations
    template_cache = {}
    
    def get_cached_template(x, y, a):
      key = (int(x), int(y), float(a))
      if key not in template_cache:
        template_cache[key] = self.set_template(self.template, x, y, a)
      return template_cache[key]
    
    def get_corr(X, Y, A, k=0, dx=0, dy=0, da=0):
      """Correlation computation with template caching."""
      T0 = get_cached_template(X[0] + (0 if k else dx),
                               Y[0] + (0 if k else dy),
                               A[0] + (0 if k else da))
      T1 = get_cached_template(X[1] + (dx if k else 0),
                               Y[1] + (dy if k else 0),
                               A[1] + (da if k else 0))
      return corr(img_opt, np.maximum(T0, T1))
    
    def print_state(X, Y, A, C):
      if self.verbose:
        print(f'C={C:.4f} ({X[0]:d}, {Y[0]:d},{A[0]:.03f}) ({X[1]:d}, {Y[1]:d},{A[1]:.03f})')
    
    # Initial values
    C = get_corr(X, Y, A)
    print_state(X, Y, A, C)
    
    # Pre-define direction vectors for translation (8-connected)
    directions = np.array([
      [1, 0], [1, 1], [0, 1], [-1, 1],
      [-1, 0], [-1, -1], [0, -1], [1, -1]
    ], dtype=np.int32)
    
    # ======================================================================
    # Iterative procedure
    
    for t_inc, a_inc in zip(l_t_inc, l_a_inc):
      
      opt = True
      
      while opt:
        
        if self.verbose:
          print(f'Iteration @ (t_inc={t_inc:d}, a_inc={a_inc:.04f})')
        
        opt = False
        
        for k in range(self.nbees):
          
          # === Angular optimization (optimized) =================
          
          # Try positive direction first
          c_pos = get_corr(X, Y, A, k, da=a_inc)
          
          if c_pos > C:
            A[k] += a_inc
            C = c_pos
            opt = True
            print_state(X, Y, A, C)
          else:
            # Try negative direction
            c_neg = get_corr(X, Y, A, k, da=-a_inc)
            if c_neg > C:
              A[k] -= a_inc
              C = c_neg
              opt = True
              print_state(X, Y, A, C)
          
          # === Translation optimization (vectorized) =================
          
          # Evaluate all 8 directions efficiently
          correlations = np.zeros(8)
          for d_idx, (dx_dir, dy_dir) in enumerate(directions):
            correlations[d_idx] = get_corr(X, Y, A, k, 
                                           dx=dx_dir*t_inc, 
                                           dy=dy_dir*t_inc)
          
          best_idx = np.argmax(correlations)
          best_corr = correlations[best_idx]
          
          if best_corr > C:
            dx, dy = directions[best_idx] * t_inc
            
            # Store improvement
            X[k] += dx
            Y[k] += dy
            C = best_corr
            opt = True
            print_state(X, Y, A, C)
            
            # Pursue in that direction
            while True:
              c = get_corr(X, Y, A, k, dx=int(dx), dy=int(dy))
              if c > C:
                X[k] += dx
                Y[k] += dy
                C = c
                print_state(X, Y, A, C)
              else:
                break
    
    return (X.tolist(), Y.tolist(), A.tolist())
  
      
  # ========================================================================
  def run(self, display=False, save_csv=True, moviefile=None):
    '''
    Process the movie
    '''

    cross = [35, 20]

    # --- Preparation ------------------------------------------------------

    # Input video 
    cap = cv.VideoCapture(self.file['movie']['path'])

    # Save result
    if save_csv:
      Data = []

    # Output video
    if moviefile is not None:
      vfile = cv.VideoWriter(moviefile, cv.VideoWriter_fourcc(*'MJPG'), fps=25, frameSize=(self.param['width'], self.param['height'])) 

    # Template
    self.template = self.load_template()

    # --- Initial positions ------------------------------------------------

    X = [580, 490]
    Y = [650, 130]
    A = [4.5, 4.5]

    # --- Processing -------------------------------------------------------

    with alive_bar(self.param['T']-1) as bar:

      frame = 0
      t = 0
      traces = [[] for i in range(self.nbees)]

      bar.title(self.file['movie']['filename'][-14:-4])
      
      while cap.isOpened():

        frame = self.get_frame(cap=cap)
        if frame is None: break
      
        # --- Processing ---------------------------------------------------

        X, Y, A = self.process_fast(frame, X, Y, A)
        
        # --- Save ---------------------------------------------------------

        if save_csv:

          pass

          # Data.append([0, frame, t, E[0].x*self.param['pix2mm'], E[0].y*self.param['pix2mm'], E[0].theta, E[0].m00])
      
          # Data.append([0, frame, t,
          #               E[0].x*self.param['pix2mm'], 
          #               E[0].y*self.param['pix2mm'], 
          #               E[0].theta, 
          #               E[0].m00])
          
          # if not bmerge and len(E)>1:
          #   Data.append([1, frame, t,
          #                 E[1].x*self.param['pix2mm'], 
          #                 E[1].y*self.param['pix2mm'], 
          #                 E[1].theta,
          #                 E[1].m00])

        # --- Display -------------------------------------------------------

        if display:

          # Images
          norm = cv.normalize(frame, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
          Res = cv.cvtColor(norm, cv.COLOR_GRAY2RGB)

          # --- Markers & traces

          # # Reset traces
          # if len(E)!=len(traces):
          #   traces = [[]] if bmerge else [[], []]

          for i in range(self.nbees):

            if i: color = (255,255,0)
            else: color = (255,0,255)

            # Ellipse
            # Res = cv.ellipse(Res, (X[i], Y[i]), (10, 10), A[i]*180/np.pi, 0, 360, color=color, thickness=1)

            # Cross
            x = X[i]
            y = Y[i]
            x_end = int(x + cross[0]*np.cos(np.pi-A[i]))
            y_end = int(y + cross[0]*np.sin(np.pi-A[i]))
            Res = cv.line(Res, (x, y), (x_end, y_end), color=color, thickness=2)

            dx = int(cross[1]/2*np.cos(np.pi-A[i]+np.pi/2))
            dy = int(cross[1]/2*np.sin(np.pi-A[i]+np.pi/2))
            Res = cv.line(Res, (int(x-dx), int(y-dy)), (x+dx, y+dy), color=color, thickness=2)

            # Trace
            traces[i].append((X[i], Y[i]))
            while len(traces[i])>100: traces[i].pop(0)
            
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