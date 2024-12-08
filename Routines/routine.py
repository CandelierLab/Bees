'''
IP routine
'''

import os
import IP

os.system('clear')

# === Parameters ===========================================================

stype = 'Social'      # 'Single' / 'Social'
btype = 'foragers'    # 'foragers' / 'nurses'

# ==========================================================================

# Data handler

H = IP.handler(stype, btype)

row = H.df.iloc[0]
# row = H.df.loc[H.df['video code'] == 'MVI_0072'].iloc[0]

print(row)
movie_code = row['video code']
dish = row['petri dish place']

pix2mm = row['pix2mm']
P = IP.processor(H.type, movie_code, dish, pix2mm=pix2mm)

# P = IP.processor(H.type, movie_code, dish)

if not os.path.exists(P.file['traj']):

  P.check_background()

  # P.run(display=True)