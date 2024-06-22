'''

Avidemux crop parameters
6x: 1420 - 580  (500x500 pix)
4x: 3180 - 1500 (660x660 pix)
'''

import os

import Analysis.Dataset.motor_activity_1B as Data
import IP

os.system('clear')

# === Parameters ===========================================================

xtype = 'x6'

# ==========================================================================

for cond, l_run in Data.groups.items():

  for run in l_run:

    print(run)

    P = IP.processor(xtype, Data.prefix + run + '.mp4')

    P.check_background()

