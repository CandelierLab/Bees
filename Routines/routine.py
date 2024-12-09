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

# row = H.df.iloc[3]
row = H.df.loc[H.df['video code'] == 'MVI_0024'].iloc[2]

movie_code = row['video code']
dish = row['petri dish place']

print(row)
print ('-'*25)

P = IP.processor(H.type, movie_code, dish)

# if not os.path.exists(P.file['traj']):
#   P.check_background()

# P.run()

P.run(display=True, save_csv=False)

# P.run(display=True, save_csv=False, moviefile='/home/raphael/Science/Projects/Misc/Bees/Movies/Social_foragers_tracking_sample_MVI_0022_2.mp4')