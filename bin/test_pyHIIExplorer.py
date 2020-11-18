#!/usr/bin/env python3

import numpy as np
import scipy as sp
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from scipy.ndimage import correlate
import matplotlib.colors as colors
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
print('Import done')
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
print(sns.__version__)

from astropy.table import QTable, Table, Column
from astropy import units as u


#
#
#
from pyHIIExplorer.HIIblob import *
from pyHIIExplorer.test import *

import time


test_print('This is a test')
quit()