'''
Manual IP 3 : trajectories
'''

import os

import IP

os.system('clear')

# === Parameters ===========================================================

season = '2025 - Summer'
stype = 'Social'      # 'Single' / 'Social'
btype = 'foragers'    # 'foragers' / 'nurses'

movie_code = 'C0001'
dish = 1

# ==========================================================================

# Data handler
H = IP.handler(season, stype, btype)

# Processor
P = IP.processor(H, movie_code, dish)

