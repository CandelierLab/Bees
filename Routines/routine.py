'''
IP routine
'''

import os
import IP

os.system('clear')

# === Parameters ===========================================================

season = '2025 - Summer'
stype = 'Social'      # 'Single' / 'Social'
btype = 'foragers'    # 'foragers' / 'nurses'

pix2mm = 0.11170251752954846

# ==========================================================================

# Data handler

H = IP.handler(season, stype, btype)

# row = H.df.iloc[3]
row = H.df.loc[H.df['video code'] == 'C0001'].iloc[2]

movie_code = row['video code']
dish = row['petri dish place']

print(row)
print ('-'*25)

P = IP.processor(H.type, movie_code, dish, pix2mm=pix2mm)

# if not os.path.exists(P.file['traj']):
#   P.check_background()

P.run(display=True, save_csv=False)

# P.run(display=True, save_csv=False, moviefile='/home/raphael/Science/Projects/Misc/Bees/Movies/Social_foragers_tracking_sample_MVI_0022_2.mp4')