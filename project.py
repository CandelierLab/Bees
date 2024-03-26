import os

# --- Root path

path = os.path.normpath(__file__)
tmp = path.split(os.sep)
root = os.sep + os.path.join(*tmp[:-3]) + os.sep

# --- Cleanup

del path, tmp