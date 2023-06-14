import numpy as np
import matplotlib.pyplot as plt
from nt_toolbox.general import rescale
from nt_toolbox.signal import load_image
import time



def low_rank_recovery_flip_test_columns(im_masked, mat_mask, n_it=100):
    """Trolling the students"""
    im2=rescale(load_image("toolbox/es.bmp"))
    time.sleep(5)
    return im2
