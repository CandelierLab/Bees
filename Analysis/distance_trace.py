import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import IP

os.system('clear')

# === Parameters ===========================================================

stype = 'Social'      # 'Single' / 'Social'
btype = 'foragers'    # 'foragers' / 'nurses'

movie_code = 'MVI_0022'
dish = 1

# ==========================================================================

# Data handler

H = IP.handler(stype, btype)

# row = H.df.iloc[0]
row = H.df.loc[H.df['video code'] == movie_code].iloc[dish-1]

print(row)
print ('-'*25)

P = IP.processor(H.type, movie_code, dish)

# Trajectory file
tfile = P.file['traj']

# Load dataframe
df = pd.read_csv(tfile)

# --- Computation

T = max(df.frame)+1

# Isplit and Imerge
Isplit = df.loc[df.id==1, 'frame'].to_numpy()
Imerge = np.setdiff1d(np.arange(T), Isplit)

# Distance
# d = np.zeros(T)
d = np.full(T, None)

for i in Isplit:
  x = df.loc[df.frame==i, 'x'].to_numpy()
  y = df.loc[df.frame==i, 'y'].to_numpy()

  d[i] = np.sqrt((np.diff(x)**2 + np.diff(y)**2)[0])

# === Display ==============================================================

# --- Compute distributions ------------------------------------------------

plt.style.use('dark_background')
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(1,1, figsize=(20,20))

ax.plot(d, '.-')

plt.show()

