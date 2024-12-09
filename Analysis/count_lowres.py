import os
import numpy as np
import pandas as pd
from alive_progress import alive_bar

import IP

os.system('clear')

# === Parameters ===========================================================

stype = 'Social'      # 'Single' / 'Social'
btype = 'foragers'    # 'foragers' / 'nurses'

# Resolution threshold
th_res = 0.125

# ==========================================================================

# Data handler
H = IP.handler(stype, btype)

# --- Conditions

match stype:

  case 'Single':
    l_cond = H.df['bee group'].unique()

  case 'Social':

    l_cond = (H.df['right wing clipped'] + '_' + H.df['left wing clipped']).unique()

#     row = H.df.iloc[0]
#   # row = H.df.loc[H.df['video code'] == 'MVI_0072'].iloc[0]

# print(row)

for k, cond in enumerate(l_cond):

  # List of runs
  match stype:
    case 'Single':
      df = H.df[H.df['bee group']==cond]
    case 'Social':
      df = H.df[(H.df['right wing clipped'] + '_' + H.df['left wing clipped'])==cond]

  num = 0

  for i, row in df.iterrows():

    # IP object
    movie_code = row['video code']
    pix2mm = row['pix2mm']

    if movie_code=='MVI_0062': 
      continue

    if pix2mm>th_res: num += 1

  print(cond, len(df))
  # print(cond, len(df), 100*num/len(df))
