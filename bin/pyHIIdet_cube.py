#!/usr/bin/python3
import numpy as np
import scipy as sp
from astropy.io import fits
from astropy import units as u
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
import matplotlib.colors as colors
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
from astropy.table import QTable, Table, Column
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from pyHIIExplorer.extract import *
from pyHIIExplorer.HIIblob import *
import warnings
import argparse

parser = argparse.ArgumentParser(description='###Program to detect HII regions from a cube###', usage='pyHIIdet_cube.py name input_file n_hdu n_Ha n_eHa FWHM_MUSE spax_sca MUSE_1sig MUSE_1sig_V plot refined maps_seg DIG_type_weight DIR [--OPTIONAL_ARGUMENTS=*]\nRun with -h for details on the inputs\n ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('name',default='galaxy',type=str, help='Name of the galaxy')
parser.add_argument('input_file', type=str, help='File of the galaxy: cube')
parser.add_argument('n_hdu', type=int, help='Index of hdu')
parser.add_argument('n_Ha', type=int, help='Index Halfa')
parser.add_argument('n_eHa', type=int, help='Index Halfa error')
parser.add_argument('FWHM_MUSE', type=float, help='FWHM of the image')
parser.add_argument('spax_sca', type=float, help='Spaxel scale')
parser.add_argument('MUSE_1sig',default=0, type=float, help='1sig, -1=eHa')
parser.add_argument('MUSE_1sig_V', default=0,type=float, help='1sig_continuum')
parser.add_argument('plot', default=0, type=int, help='Plot, 0=not 1=yes')
parser.add_argument('refined',default=0, type=int, help='Refined detection')
parser.add_argument('maps_seg', default=0, type=int, help='To do segmentation maps, 0=not 1=yes')
parser.add_argument('DIG_type_weight', default=0, type=int, help='Create new DIG 0=not 1=yes')
parser.add_argument('DIR', default='none', type=str, help='Where save outputfiles')

args = parser.parse_args()

print('Reading files')

name = args.name
input_file = args.input_file
n_hdu = args.n_hdu
n_Ha = args.n_Ha
n_eHa = args.n_eHa
FWHM_MUSE = args.FWHM_MUSE
spax_sca = args.spax_sca
MUSE_1sig = args.MUSE_1sig
MUSE_1sig_V = args.MUSE_1sig_V
plot = args.plot
refined = args.refined
maps_seg = args.maps_seg
DIG_type_weight = args.DIG_type_weight
DIR = args.DIR

plt.ion()

hdu = fits.open(input_file)

data = hdu[n_hdu].data
hdr_fe = hdu[n_hdu].header
F_Ha_MUSE = data[n_Ha,:,:]
eF_Ha_MUSE = data[n_eHa,:,:]
nx=hdr_fe["NAXIS1"]
ny=hdr_fe["NAXIS2"]
nz=hdr_fe["NAXIS3"]

FWHM_MUSE = FWHM_MUSE/spax_sca
V_MUSE = gaussian_filter(F_Ha_MUSE, sigma=FWHM_MUSE)
V_MUSE = np.ma.masked_invalid(V_MUSE)

print('Detecting HII regions')

if(MUSE_1sig == -1.0):
    
    MUSE_1sig = np.nanmedian(eF_Ha_MUSE[F_Ha_MUSE!=0])
    print(MUSE_1sig)

if(MUSE_1sig_V == -1.0):
    
    eV_MUSE = gaussian_filter(eF_Ha_MUSE, sigma=FWHM_MUSE)
    mean_MUSE_1sig_V = np.nanmean(eV_MUSE)
    median_MUSE_1sig_V = np.nanmedian(eV_MUSE)
    MUSE_1sig_V = np.min(np.array([mean_MUSE_1sig_V, median_MUSE_1sig_V]))

    if(np.isnan(MUSE_1sig_V)):
        MUSE_1sig_V = MUSE_1sig
    
blobs_final,blobs_F_Ha,image_HII,diff_map_final,diff_points,diff_Flux=HIIblob(F_Ha_MUSE,V_MUSE,FWHM_MUSE,MUSE_1sig=MUSE_1sig, MUSE_1sig_V=MUSE_1sig_V, plot=plot, refined=refined, name=name)

if (DIG_type_weight==1):
    Ha_image_clean = F_Ha_MUSE - image_HII
    diff_map_final =  create_diff_new(Ha_image_clean,blobs_final,FWHM_MUSE,diff_points)
    
print('Creating dictionaries with HII regions and DIG')

dict_HII = {}
list_HII = []
dict_HII_units = {}

for i in range(len(blobs_final)):
    list_HII.append(name+'-'+str(i+1))
dict_HII['HIIREGID'] = list_HII
dict_HII['X'] = blobs_final[:,1]
dict_HII['Y'] = blobs_final[:,0]
dict_HII['R'] = blobs_final[:,2]
dict_HII['flux'] = blobs_F_Ha
dict_HII_units['HIIREGID'] = 'none'
dict_HII_units['X'] = 'spaxels'
dict_HII_units['Y'] = 'spaxels'
dict_HII_units['R'] = 'spaxels'
dict_HII_units['flux'] = '10^-16 erg/s/cm^2'


dict_DIG = {}
list_DIG = []
dict_DIG_units = {}

for i in range(len(diff_points)):
    list_DIG.append(name+'-DIG-'+str(i+1))
dict_DIG['DIGREGID'] = list_DIG
dict_DIG['X'] = diff_points[:,0]
dict_DIG['Y'] = diff_points[:,1]
dict_DIG['flux'] = diff_Flux
dict_DIG_units['DIGREGID'] = 'none'
dict_DIG_units['X'] = 'spaxels'
dict_DIG_units['Y'] = 'spaxels'
dict_DIG_units['flux'] = '10^-16 erg/s/cm^2'

print('Saving and writing outputfiles')

table_HII = Table(dict_HII)
for key in dict_HII.keys():
    table_HII[key].unit=dict_HII_units[key]

file_table_HII = DIR+'/'+name+'.HIIblob_HII.tab.ecsv'
table_HII.write(file_table_HII, overwrite=True, delimiter=',')  
  

table_DIG = Table(dict_DIG)
for key in dict_DIG.keys():
    table_DIG[key].unit=dict_DIG_units[key]

file_table_DIG = DIR+'/'+name+'.HIIblob_DIG.tab.ecsv'
table_DIG.write(file_table_DIG, overwrite=True, delimiter=',')  
  

hdu_HII = fits.PrimaryHDU(data = image_HII)
hdu_HII.writeto(DIR+'/'+name+"_image_HII.fits", output_verify="ignore", overwrite=True)
  
hdu_DIG = fits.PrimaryHDU(data = diff_map_final)
hdu_DIG.writeto(DIR+'/'+name+"_image_DIG.fits", output_verify="ignore",overwrite=True)

blobs=blobs_final
  
if (maps_seg==1):

    print('Create segmentation maps')
    map_seg=create_HII_seg(blobs, nx, ny)
    map_mask_seg=create_mask_HII_seg(blobs, nx, ny)
    
    print('Saving and writing output files')
    hdu_map_seg = fits.PrimaryHDU(data = map_seg)
    hdu_map_seg.header['OBJECT']=name
    hdu_map_seg.writeto(DIR+'/'+name+"_map_seg.fits", output_verify="ignore", overwrite=True)
    
    hdu_map_mask_seg = fits.PrimaryHDU(data = map_mask_seg )
    hdu_map_mask_seg.header['OBJECT']=name
    hdu_map_mask_seg.writeto(DIR+'/'+name+"_map_mask_seg.fits", output_verify="ignore", overwrite=True)


print("End of program")







