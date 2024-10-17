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
btype = 'foragers'    # 'foragers' / 'nurses'

movie_code = 'MVI_0038'
dish = 1

tau = 1
vbin = np.linspace(0, 100, 200)

# ==========================================================================

# Data handler
H = IP.handler(stype, btype)

# IP object
P = IP.processor(H.type, movie_code, dish)

# Trajectory file
tfile = P.file['traj']

# Skip if not existing
if not os.path.exists(tfile):
  print('No file', movie_code, dish)

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

# === Display ==============================================================

# --- Compute distributions ------------------------------------------------

# plt.style.use('dark_background')
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(1,2, figsize=(40,20))

ax[0].plot(x, 'r.')
ax[0].plot(y, 'b.')

ax[0].plot(x_, 'r-', label='x (mm)')
ax[0].plot(y_, 'b-', label='y (mm)')
ax[0].plot(v, 'm-', label='v (mm/s)')

ax[0].set_xlim(0, 500)
ax[0].set_ylim(0, 100)
ax[0].set_xlabel('t (frames)')
ax[0].legend()

pdf = stats.gaussian_kde(v, bw_method=0.01)
ax[1].plot(vbin, pdf.evaluate(vbin), '-')

ax[1].set_ylim(1e-4, 1)

ax[1].set_yscale('log')
ax[1].set_xlabel('speed (mm/s)')
ax[1].set_ylabel('pdf')

plt.show()

