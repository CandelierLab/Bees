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
btype = 'nurses'    # 'foragers' / 'nurses'

tau = 1
vbin = np.linspace(0, 100, 200)

# ==========================================================================

# Data handler
H = IP.handler(stype, btype)

# Conditions
l_cond = H.df['bee group'].unique()

# Containers
speed = {}

for cond in l_cond:

  # List of runs
  df = H.df[H.df['bee group']==cond]
  
  # Speed container
  speed[cond] = np.empty(0)

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
      x = df.x.to_numpy()
      y = df.y.to_numpy()
      t = df.t.to_numpy()

      # Filtering
      x_ = savgol_filter(x, 51, 3)
      y_ = savgol_filter(y, 51, 3)

      dx = x_[tau:] - x_[0:-tau]
      dy = y_[tau:] - y_[0:-tau]
      dr = np.sqrt(dx**2 + dy**2)
      dt = t[tau:] - t[0:-tau]

      v = dr/dt

      # Storage
      speed[cond] = np.concatenate((speed[cond], v))

      bar()

# === Display ==============================================================

# --- Compute distributions ------------------------------------------------

# plt.style.use('dark_background')
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots()

K = []
for cond, v in speed.items():

  pdf = stats.gaussian_kde(speed[cond], bw_method=0.01)

  ax.plot(vbin, pdf.evaluate(vbin), '-', label=cond)

  # break

ax.set_ylim(1e-4, 1)

ax.set_yscale('log')
ax.set_xlabel('speed (mm/s)')
ax.set_ylabel('pdf')

ax.legend()

plt.show()

