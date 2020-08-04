from skimage.io import imread
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np

from whole_leg.tibia.ankle import calc_ankle_level

mask = imread("")

mask = mask.astype(np.uint8).squeeze()
mask = binary_fill_holes(mask)

y, x = calc_ankle_level(mask)
print("")
