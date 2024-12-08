'''
Batch run on a whole condition
'''


import os
import IP

os.system('clear')

# === Parameters ===========================================================

stype = 'Social'      # 'Single' / 'Social'
btype = 'nurses'      # 'foragers' / 'nurses'

# ==========================================================================

# Data handler
H = IP.handler(stype, btype)

for index, row in H.df.iterrows():

  movie_code = row['video code']
  dish = row['petri dish place']

  if movie_code=='MVI_0062': 
    continue

  print(movie_code, dish)

  pix2mm = row['pix2mm']
  P = IP.processor(H.type, movie_code, dish, pix2mm=pix2mm)

  # P = IP.processor(H.type, movie_code, dish)

  # Background extraction
  P.check_background()

  # if not os.path.exists(P.file['traj']):
  #   P.run(display=False)