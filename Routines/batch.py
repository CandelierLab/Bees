'''

Avidemux crop parameters
6x: 1420 - 580  (500x500 pix)
4x: 3180 - 1500 (660x660 pix)
'''

import os

import Analysis.Dataset.motor_activity_1B as Data
import IP

os.system('clear')

for cond, l_run in Data.groups.items():

  for run in l_run:

    print(run)
    P = IP.processor(Data.prefix + run + '.mp4', verbose=False)

    if not os.path.exists(P.file['traj']):
      P.run(display=True)

