import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import project

os.system('clear')

# === Parameters ===========================================================

tag = 'sNPF videos -  Motor activity/14062024/MVI_0012/MVI_0012_2'

# --------------------------------------------------------------------------

tfile = project.root + 'Files/' + tag + '/trajectories.csv'

# ==========================================================================

df = pd.read_csv(tfile)

# plt.style.use('dark_background')
ax = plt.axes(projection='3d')

ax.plot3D(df.x, df.y, df.t, '-')

plt.show()

