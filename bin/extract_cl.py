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
#from pyHIIExplorer.extract import *
from pyHIIExplorer.HIIblob import *
from extract import *
import warnings
import argparse


parser = argparse.ArgumentParser(description='###Program to extract properties of HII regions and DIG from separate input files###', usage='extract_cl.py name input_file_fe input_file_ssp input_file_sfh input_file_index blobs_final diff_points FWHM_MUSE plot def_DIG DIR [--OPTIONAL_ARGUMENTS=*]\nRun with -h for details on the inputs\n ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


#n_hdu_fe n_hdu_ssp n_hdu_sfh n_hdu_index



parser.add_argument('name', type=str, help='Name of the galaxy')
parser.add_argument('input_file_fe', type=str, help='Input file flux elines')
parser.add_argument('input_file_ssp', type=str, help='Input file ssp')
parser.add_argument('input_file_sfh', type=str, help='Input file sfh')
parser.add_argument('input_file_index', type=str, help='Input file indices')
#parser.add_argument('n_hdu_fe', type=int, help='hdu flux_elines')
#parser.add_argument('n_hdu_ssp', type=int, help='hdu ssp')
#parser.add_argument('n_hdu_sfh', type=int, help='hdu_sfh')
#parser.add_argument('n_hdu_index', type=int, help='hdu_index')
parser.add_argument('blobs_final', type=str, help='Blobs of detection HII regions (table)')
parser.add_argument('diff_points', type=str, help='Points diffuse (table)')
parser.add_argument('FWHM_MUSE', type=float, help='FWHM')
parser.add_argument('plot', type=int, help='Plot, 0=not 1=yes')
parser.add_argument('def_DIG', default=0, type=int, help='New diffuse, 0=not, 1=yes')
parser.add_argument('DIR', default='none', type=str, help='Where save outputfiles')

args = parser.parse_args()

name = args.name
input_file_fe = args.input_file_fe
input_file_ssp = args.input_file_ssp
input_file_sfh = args.input_file_sfh
input_file_index = args.input_file_index
#n_hdu_fe = args.n_hdu_fe
#n_hdu_ssp = args.n_hdu_ssp
#n_hdu_sfh = args.n_hdu_sfh
#n_hdu_index = args.n_hdu_index
blobs_final = args.blobs_final
diff_points = args.diff_points
FWHM_MUSE = args.FWHM_MUSE
plot = args.plot
def_DIG = args.def_DIG
DIR = args.DIR

plt.ion()

hdu = fits.open(input_file_fe)

data = hdu[0].data
hdr_fe = hdu[0].header

hdu_ssp = fits.open(input_file_ssp)
hdu_sfh = fits.open(input_file_sfh)
hdu_index = fits.open(input_file_index)

#data = hdu[n_hdu_fe].data
#hdr_fe = hdu[n_hdu_fe].header

#data_SSP = hdu[n_hdu_ssp].data
#hdr_ssp = hdu[n_hdu_ssp].header

#data_SFH = hdu[n_hdu_sfh].data
#hdr_sfh = hdu[n_hdu_sfh].header

#data_INDEX = hdu[n_hdu_index].data
#hdr_index = hdu_[n_hdu_index].header


blobs_final = Table.read(blobs_final, format='fits')
diff_points = Table.read(diff_points, format='fits')


blobs_final = np.array([blobs_final['Y'], blobs_final['X'], blobs_final['R']])
blobs_final = blobs_final.transpose()

diff_points = np.array([diff_points['X'], diff_points['Y']])
diff_points = diff_points.transpose()


WCS, hdu_HII, hdu_DIG, table_HII, table_DIG = extracting_flux_elines(name, hdu, blobs_final, diff_points,  plot=plot)

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

    WCS, hdu_HII, hdu_DIG, table_HII, table_DIG = extracting_flux_elines(name, hdu, blobs_final, diff_points,  plot=plot, def_DIG=def_DIG,cube_DIG=data_DIG)
    
    hdu_DIG.data = data_DIG

print('Saving outputsfile flux_elines')

file_table_HII = DIR+"/HII."+name+".flux_elines.table.fits.gz"
file_table_DIG = DIR+"/DIG."+name+".flux_elines.table.fits.gz"

file_HII = DIR+"/HII."+name+".flux_elines.cube.fits.gz"
file_DIG = DIR+"/DIG."+name+".flux_elines.cube.fits.gz"


table_HII.write(file_table_HII, overwrite=True)
table_DIG.write(file_table_DIG, overwrite=True)

hdu_HII.writeto(file_HII,  overwrite=True)
hdu_DIG.writeto(file_DIG, overwrite=True)

print('Extracting SSP')

hdu_SSP_HII, hdu_SSP_DIG, table_SSP_HII, table_SSP_DIG = extracting_ssp(name, hdu_ssp, WCS, 
                                                                                    blobs_final, diff_points, plot=plot)

print('Saving outputsfile SSP')

file_table_SSP_HII = DIR+"/HII."+name+'.SSP.table.fits.gz'
file_table_SSP_DIG = DIR+"/DIG."+name+'.SSP.table.fits.gz'

file_SSP_HII = DIR+"/HII."+name+'.SSP.cube.fits.gz'
file_SSP_DIG = DIR+"/DIG."+name+'.SSP.cube.fits.gz'

table_SSP_HII.write(file_table_SSP_HII, overwrite=True)
table_SSP_DIG.write(file_table_SSP_DIG, overwrite=True)

hdu_SSP_HII.writeto(file_SSP_HII, overwrite=True)
hdu_SSP_DIG.writeto(file_SSP_DIG, overwrite=True)

print('Extracting SFH')

hdu_SFH_HII, hdu_SFH_DIG, table_SFH_HII, table_SFH_DIG = extracting_sfh(name, hdu_sfh, WCS, 
                                                                                    blobs_final, diff_points, plot=plot)
                                                                                    
print('Saving outputsfile SFH')

file_table_SFH_HII = DIR+"/HII."+name+'.SFH.table.fits.gz'
file_table_SFH_DIG = DIR+"/DIG."+name+'.SFH.table.fits.gz'

file_SFH_HII = DIR+"/HII."+name+'.SFH.cube.fits.gz'
file_SFH_DIG = DIR+"/DIG."+name+'.SFH.cube.fits.gz'

table_SFH_HII.write(file_table_SFH_HII, overwrite=True)
table_SFH_DIG.write(file_table_SFH_DIG, overwrite=True)

hdu_SFH_HII.writeto(file_SFH_HII, overwrite=True)
hdu_SFH_DIG.writeto(file_SFH_DIG, overwrite=True)

print('Extracting INDEX')
                                                                                    
hdu_INDEX_HII, hdu_INDEX_DIG, table_INDEX_HII, table_INDEX_DIG = extracting_index(name, hdu_index, WCS, 
                                                                                    blobs_final, diff_points, plot=plot)    

print('Saving outputsfile INDEX')

file_table_INDEX_HII = DIR+"/HII."+name+'.INDEX.table.fits.gz'
file_table_INDEX_DIG = DIR+"/DIG."+name+'.INDEX.table.fits.gz'

file_INDEX_HII = DIR+"/HII."+name+'.INDEX.cube.fits.gz'
file_INDEX_DIG = DIR+"/DIG."+name+'.INDEX.cube.fits.gz'
        
table_INDEX_HII.write(file_table_INDEX_HII, overwrite=True)
table_INDEX_DIG.write(file_table_INDEX_DIG, overwrite=True)     

hdu_INDEX_HII.writeto(file_INDEX_HII, output_verify="ignore", overwrite=True)
hdu_INDEX_DIG.writeto(file_INDEX_DIG, output_verify="ignore", overwrite=True)  


print('End program')








