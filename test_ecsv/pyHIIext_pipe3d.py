#!/usr/bin/python3
import numpy as np
import scipy as sp
from astropy.io import fits
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
from pyHIIExplorer.extract_all import *
from pyHIIExplorer.HIIblob import *
import warnings
import argparse

parser = argparse.ArgumentParser(description='###Program to extract properties of HII regions and DIG from cube pip3d###', usage='pyHIIext_pipe3d.py name input_file n_hdu_fe n_hdu_ssp n_hdu_sfh n_hdu_index blobs_final diff_points FWHM_MUSE spax_sca plot def_DIG DIR [--OPTIONAL_ARGUMENTS=*]\nRun with -h for details on the inputs\n ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('name', type=str, help='Name of the galaxy')
parser.add_argument('input_file', type=str, help='File of the galaxy: cube pipe3d')
parser.add_argument('n_hdu_fe', type=int, help='hdu flux_elines')
parser.add_argument('n_hdu_ssp', type=int, help='hdu ssp')
parser.add_argument('n_hdu_sfh', type=int, help='hdu_sfh')
parser.add_argument('n_hdu_index', type=int, help='hdu_index')
parser.add_argument('blobs_final', type=str, help='Blobs of detection HII regions (table)')
parser.add_argument('diff_points', type=str, help='Points diffuse (table)')
parser.add_argument('FWHM_MUSE', type=float, help='FWHM')
parser.add_argument('spax_sca', type=float, help='Spaxel scale')
parser.add_argument('plot', type=int, help='Plot, 0=not 1=yes')
parser.add_argument('def_DIG', default=0, type=int, help='New diffuse, 0=not, 1=yes')
parser.add_argument('DIR', default='none', type=str, help='Where save outputfiles')

args = parser.parse_args()

print('Reading files')

name = args.name
input_file = args.input_file
n_hdu_fe = args.n_hdu_fe
n_hdu_ssp = args.n_hdu_ssp
n_hdu_sfh = args.n_hdu_sfh
n_hdu_index = args.n_hdu_index
blobs_final = args.blobs_final
diff_points = args.diff_points
FWHM_MUSE = args.FWHM_MUSE
spax_sca = args.spax_sca
plot = args.plot
def_DIG = args.def_DIG
DIR = args.DIR

plt.ion()

hdu = fits.open(input_file)

data = hdu[n_hdu_fe].data
hdr_fe = hdu[n_hdu_fe].header
FWHM_MUSE = FWHM_MUSE/spax_sca

data_SSP = hdu[n_hdu_ssp].data
hdr_ssp = hdu[n_hdu_ssp].header

data_SFH = hdu[n_hdu_sfh].data
hdr_sfh = hdu[n_hdu_sfh].header

data_INDEX = hdu[n_hdu_index].data
hdr_index = hdu[n_hdu_index].header

blobs_final = Table.read(blobs_final)
diff_points = Table.read(diff_points)


blobs_final = np.array([blobs_final['Y'], blobs_final['X'], blobs_final['R']])
blobs_final = blobs_final.transpose()

diff_points = np.array([diff_points['X'], diff_points['Y']])
diff_points = diff_points.transpose()

print('Extracting flux_elines')

WCS, hdu_HII, hdu_DIG, table_HII, table_DIG = extracting_flux_elines(name, data, hdr_fe, blobs_final, diff_points, FWHM_MUSE,  plot=plot)

if(def_DIG==1):
    
    print('Extracting new DIG')
    
    data_DIG = hdu_DIG.data
    hdr_fe_DIG = hdu_DIG.header

    data_HII = hdu_HII.data
    hdr_fe_HII = hdu_HII.header
    
    cube_clean = data-data_HII
    cube_diff = create_diff_cube(cube_clean,blobs_final,FWHM_MUSE,diff_points)
    nz = hdr_fe['NAXIS3']
    nz_flux = int(nz/8)
    data_DIG[:nz_flux,:,:] = cube_diff[:nz_flux,:,:]

    WCS, hdu_HII, hdu_DIG, table_HII, table_DIG = extracting_flux_elines(name, data, hdr_fe, blobs_final, diff_points, FWHM_MUSE,  plot=plot, def_DIG=def_DIG,cube_DIG=data_DIG)
    
    hdu_DIG.data = data_DIG

print('Saving outputsfile flux_elines')

file_table_HII = DIR+'/HII.'+name+'.flux_elines.table.ecsv'
file_table_DIG = DIR+'/DIG.'+name+'.flux_elines.table.ecsv'

file_HII = DIR+'/HII.'+name+'.flux_elines.cube.fits.gz'
file_DIG = DIR+'/DIG.'+name+'.flux_elines.cube.fits.gz'

table_HII.write(file_table_HII, overwrite=True, delimiter=',')
table_DIG.write(file_table_DIG, overwrite=True, delimiter=',')

hdu_HII.writeto(file_HII,  overwrite=True)
hdu_DIG.writeto(file_DIG, overwrite=True)

print('Extracting SSP')

hdu_SSP_HII, hdu_SSP_DIG, table_SSP_HII, table_SSP_DIG = extracting_ssp(name, data_SSP, hdr_ssp, WCS, 
                                                                                    blobs_final, diff_points, FWHM_MUSE, plot=plot)

print('Saving outputsfile SSP')

file_table_SSP_HII = DIR+'/HII.'+name+'.SSP.table.ecsv'
file_table_SSP_DIG = DIR+'/DIG.'+name+'.SSP.table.ecsv'

file_SSP_HII = DIR+'/HII.'+name+'.SSP.cube.fits.gz'
file_SSP_DIG = DIR+'/DIG.'+name+'.SSP.cube.fits.gz'

table_SSP_HII.write(file_table_SSP_HII, overwrite=True, delimiter=',')
table_SSP_DIG.write(file_table_SSP_DIG, overwrite=True, delimiter=',')

hdu_SSP_HII.writeto(file_SSP_HII, overwrite=True)
hdu_SSP_DIG.writeto(file_SSP_DIG, overwrite=True)

print('Extracting SFH')

hdu_SFH_HII, hdu_SFH_DIG, table_SFH_HII, table_SFH_DIG = extracting_sfh(name, data_SFH, hdr_sfh, WCS, 
                                                                                    blobs_final, diff_points, FWHM_MUSE, plot=plot)
                                                                                    
print('Saving outputsfile SFH')

file_table_SFH_HII = DIR+'/HII.'+name+'.SFH.table.ecsv'
file_table_SFH_DIG = DIR+'/DIG.'+name+'.SFH.table.ecsv'

file_SFH_HII = DIR+'/HII.'+name+'.SFH.cube.fits.gz'
file_SFH_DIG = DIR+'/DIG.'+name+'.SFH.cube.fits.gz'

table_SFH_HII.write(file_table_SFH_HII, overwrite=True, delimiter=',')
table_SFH_DIG.write(file_table_SFH_DIG, overwrite=True, delimiter=',')

hdu_SFH_HII.writeto(file_SFH_HII, overwrite=True)
hdu_SFH_DIG.writeto(file_SFH_DIG, overwrite=True)

print('Extracting INDEX')
                                                                                    
hdu_INDEX_HII, hdu_INDEX_DIG, table_INDEX_HII, table_INDEX_DIG = extracting_index(name, data_INDEX, hdr_index, WCS, 
                                                                                    blobs_final, diff_points, FWHM_MUSE, plot=plot)    

print('Saving outputsfile INDEX')

file_table_INDEX_HII = DIR+'/HII.'+name+'.INDEX.table.ecsv'
file_table_INDEX_DIG = DIR+'/DIG.'+name+'.INDEX.table.ecsv'

file_INDEX_HII = DIR+'/HII.'+name+'.INDEX.cube.fits.gz'
file_INDEX_DIG = DIR+'/DIG.'+name+'.INDEX.cube.fits.gz'
        
table_INDEX_HII.write(file_table_INDEX_HII, overwrite=True, delimiter=',')
table_INDEX_DIG.write(file_table_INDEX_DIG, overwrite=True, delimiter=',')     

hdu_INDEX_HII.writeto(file_INDEX_HII, output_verify='ignore', overwrite=True)
hdu_INDEX_DIG.writeto(file_INDEX_DIG, output_verify='ignore', overwrite=True)  

print('End program')








