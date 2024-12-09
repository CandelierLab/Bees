'''
Batch run on a whole condition
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

for index, row in H.df.iterrows():

  movie_code = row['video code']
  dish = row['petri dish place']

  P = IP.processor(H.type, movie_code, dish)

  # Background extraction
  # P.check_background()

  if not os.path.exists(P.file['traj']):
    P.run()