import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import project

import Analysis.Dataset.motor_activity_1B as dataset

os.system('clear')

# === Parameters ===========================================================

# tfile = project.root + 'Files/' + tag + '/trajectories.csv'

# ==========================================================================

print(dataset.prefix)

l_cond = dataset.groups.keys()

for cond in l_cond:

  l_run = dataset.groups[cond]

  for run in l_run:

    # Trajectory file
    tfile = project.root + 'Files/' + dataset.prefix + run + '/trajectories.csv'

    # Skip if not existing
    if not os.path.exists(tfile):
      continue

    # Dataframe
    df = pd.read_csv(tfile)

    # Compute distance

    dr = np.sqrt(np.diff(df.x)**2 + np.diff(df.y)**2)

    break

  break



plt.style.use('dark_background')
fig, ax = plt.subplots()

ax.plot(dr, '.-')

plt.show()

