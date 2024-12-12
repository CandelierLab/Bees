import os
import numpy as np
from scipy import stats
import pandas as pd
from scipy.signal import savgol_filter
from alive_progress import alive_bar
import matplotlib.pyplot as plt

import IP

os.system('clear')

# === Parameters ===========================================================

stype = 'Single'      # 'Single' / 'Social'
btype = 'nurses'      # 'foragers' / 'nurses'

th_dr = 1

# ==========================================================================

# Data handler
H = IP.handler(stype, btype)

# Conditions
l_cond = H.df['bee group'].unique()

# Containers
speed = []

for k, cond in enumerate(l_cond):

  # List of runs
  df = H.df[H.df['bee group']==cond]
  
  # Speed container
  speed.append([])

  with alive_bar(df.shape[0]) as bar:

    bar.title(cond)

    for i, row in df.iterrows():

      # IP object
      movie_code = row['video code']
      dish = row['petri dish place']
      P = IP.processor(H.type, movie_code, dish)

      # Trajectory file
      tfile = P.file['traj']

      # Skip if not existing
      if not os.path.exists(tfile):
        print('No file', movie_code, dish)
        continue

      # Load dataframe
      df = pd.read_csv(tfile)

      # -- Compute and store speed

      # Computation
      X = df.x.to_numpy()
      Y = df.y.to_numpy()
      T = df.t.to_numpy()

      # Filtering
      x = savgol_filter(X, 51, 3)
      y = savgol_filter(Y, 51, 3)

      iref = 0
      D = 0
      for i, t in enumerate(T):

        # Distance to reference
        dr2 = (x[i]-x[iref])**2 + (y[i]-y[iref])**2

        # Update
        if dr2 >= th_dr or i==T.size-1:
          D += np.sqrt(dr2)
          iref = i

      # Storage
      speed[k].append(np.log10(D/max(T)))

      bar()

# === Display ==============================================================

# --- Compute distributions ------------------------------------------------

# plt.style.use('dark_background')
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(1,1, figsize=(20,20))

ax.violinplot(speed, showmeans=True)

for k, cond in enumerate(l_cond):
  n = len(speed[k])
  x = (k+1)*np.ones(n) + 0.02*np.random.randn(n)
  ax.scatter(x, speed[k])

ax.set_xticks([y + 1 for y in range(len(l_cond))],
                  labels=l_cond)

ax.set_ylabel('Speed $log_{10}(v)$  (mm/s)')
ax.set_title(H.type)

plt.show()

