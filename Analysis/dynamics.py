import os
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import project

import Analysis.Dataset.motor_activity_1B as dataset

os.system('clear')

# === Parameters ===========================================================

# tfile = project.root + 'Files/' + tag + '/trajectories.csv'

# ==========================================================================

# Conditions
l_cond = dataset.groups.keys()

# Containers
speed = {}

for cond in l_cond:

  # List of runs
  l_run = dataset.groups[cond]

  # Speed container
  speed[cond] = np.empty(0)

  for run in l_run:

    # --- Load data

    # Trajectory file
    tfile = project.root + 'Files/' + dataset.prefix + run + '/trajectories.csv'

    # Skip if not existing
    if not os.path.exists(tfile):
      continue

    # Load dataframe
    df = pd.read_csv(tfile)

    # -- Compute and store speed

    # Computation
    dr = np.sqrt(np.diff(df.x)**2 + np.diff(df.y)**2)
    dt = np.diff(df.t)
    v = dr/dt

    # Storage
    speed[cond] = np.concatenate(speed[cond], v)

    break

  break

# === Display ==============================================================

# --- Compute distributions ------------------------------------------------



# --- Figure ---------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots()

ax.plot(dr, '.-')

plt.show()

