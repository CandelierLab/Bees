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

stype = 'Social'        # 'Single' / 'Social'
btype = 'nurses'      # 'foragers' / 'nurses'

th_dr = 1

# plot = 'violin'
plot = 'table'

# ==========================================================================

# Data handler
H = IP.handler(stype, btype)

# --- Conditions
tmp = pd.concat([H.df['left wing clipped'] + '_' + H.df['right wing clipped'],
                    H.df['right wing clipped'] + '_' + H.df['left wing clipped']]).unique()

l_cond = []
l_state = []
for c in tmp:
  s = c.split('_')
  l_state.append(s[0])
  l_state.append(s[1])
  l_cond.append(c if s[0]<s[1] else f'{s[1]}_{s[0]}')

# Unique values
l_cond = list(set(l_cond))
l_state = list(set(l_state))
l_cond.sort()
l_state.sort()

ns = len(l_state)

# --- Data aggregation

# Containers
dist = []
md = np.full((ns, ns), np.nan)

for k, cond in enumerate(l_cond):
 
  # List of runs
  df = pd.concat([H.df[(H.df['right wing clipped'] + '_' + H.df['left wing clipped'])==cond],
                  H.df[(H.df['left wing clipped'] + '_' + H.df['right wing clipped'])==cond]])
  
  dist.append([])

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

      # -- Computation

      T = max(df.frame)+1

      # Isplit and Imerge
      Isplit = df.loc[df.id==1, 'frame'].to_numpy()
      # Imerge = np.setdiff1d(np.arange(T), Isplit)

      if len(Isplit):

        # Distance
        d = np.full(T, np.nan)

        for i in Isplit:
          x = df.loc[df.frame==i, 'x'].to_numpy()
          y = df.loc[df.frame==i, 'y'].to_numpy()

          d[i] = np.sqrt((np.diff(x)**2 + np.diff(y)**2)[0])

        dist[k].append(np.nanmean(d))
      
      bar()

  # --- Mean values

  # States
  s = cond.split('_')
  i = l_state.index(s[0])
  j = l_state.index(s[1])

  md[i,j] = np.mean(dist[k])
  md[j,i] = np.mean(dist[k])

# === Display ==============================================================

# plt.style.use('dark_background')
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(1,1, figsize=(20, 15))

match plot:

  case 'violin':

    bins = np.arange(len(l_cond))+1

    ax.violinplot(dist, showmeans=True)

    for k, cond in enumerate(l_cond):
      n = len(dist[k])
      x = (k+1)*np.ones(n) + 0.02*np.random.randn(n)
      ax.scatter(x, dist[k])

    ax.set_xticks(bins, labels=l_cond, rotation=30, ha='right')

    ax.set_ylabel('Distance (mm)')

    ax.set_title(H.type)

  case 'table':

    cmap = plt.colormaps.get_cmap('turbo')
    cmap.set_bad(color='white')

    ms = ax.matshow(md, cmap=cmap, vmin=0, vmax=50)

    ax.set_xticks(range(ns), labels=l_state)
    ax.set_yticks(range(ns), labels=l_state)

    fig.colorbar(ms, ax=ax)

    ax.set_title(H.type, y=1.1)


plt.show()

