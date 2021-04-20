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
from pyHIIExplorer.HIIblob import *
from pyHIIExplorer.extract_all import *
import warnings
import argparse

parser = argparse.ArgumentParser(description='###Program to detect and extract HII regions and DIG###', usage='pyHIIdet_ext_pipe3d.py name input_file n_hdu_fe n_hdu_ssp n_hdu_sfh n_hdu_index n_Ha n_eHa FWHM_MUSE spax_sca MUSE_1sig MUSE_1sig_V plot refined maps_seg def_DIG DIR [--OPTIONAL_ARGUMENTS=*]\nRun with -h for details on the inputs\n ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('name',default='galaxy',type=str, help='Name of the galaxy')
parser.add_argument('input_file', type=str, help='File of the galaxy: cube pip3d')
parser.add_argument('n_hdu_fe', type=int, help='Index of hdu_fe')
parser.add_argument('n_hdu_ssp', type=int, help='Index of hdu_ssp')
parser.add_argument('n_hdu_sfh', type=int, help='Index of hdu_sfh')
parser.add_argument('n_hdu_index', type=int, help='Index of hdu_index')
parser.add_argument('n_Ha', type=int, help='Index Halfa')
parser.add_argument('n_eHa', type=int, help='Index Halfa error')
parser.add_argument('FWHM_MUSE', type=float, help='FWHM of the image')
parser.add_argument('spax_sca', type=float, help='Spaxel scale')
parser.add_argument('MUSE_1sig',default=0, type=float, help='1sig')
parser.add_argument('MUSE_1sig_V', default=0,type=float, help='1sig_continuum')
parser.add_argument('plot', default=0, type=int, help='Plot')
parser.add_argument('refined',default=0, type=int, help='Refined detection')
parser.add_argument('maps_seg', default=0, type=int, help='To do segmentation maps')
parser.add_argument('def_DIG', default=0, type=int, help='Create new DIG')
parser.add_argument('DIR', default='none', type=str, help='Where save outputfiles')

args = parser.parse_args()

print('Reading files')

name = args.name
input_file = args.input_file
n_hdu_fe = args.n_hdu_fe
n_hdu_ssp = args.n_hdu_ssp
n_hdu_sfh = args.n_hdu_sfh
n_hdu_index = args.n_hdu_index
n_Ha = args.n_Ha
n_eHa = args.n_eHa
FWHM_MUSE = args.FWHM_MUSE
spax_sca = args.spax_sca
MUSE_1sig = args.MUSE_1sig
MUSE_1sig_V = args.MUSE_1sig_V
plot = args.plot
refined = args.refined
maps_seg = args.maps_seg
def_DIG = args.def_DIG
DIR = args.DIR

plt.ion()

hdu = fits.open(input_file)

data = hdu[n_hdu_fe].data
hdr_fe = hdu[n_hdu_fe].header

data_SSP = hdu[n_hdu_ssp].data
hdr_ssp = hdu[n_hdu_ssp].header

data_SFH = hdu[n_hdu_sfh].data
hdr_sfh = hdu[n_hdu_sfh].header

data_INDEX = hdu[n_hdu_index].data
hdr_index = hdu[n_hdu_index].header

F_Ha_MUSE = data[n_Ha,:,:]
eF_Ha_MUSE = data[n_eHa,:,:]
nx = hdr_fe["NAXIS1"]
ny = hdr_fe["NAXIS2"]
nz = hdr_fe["NAXIS3"]

FWHM_MUSE = FWHM_MUSE/spax_sca
V_MUSE = gaussian_filter(F_Ha_MUSE, sigma=FWHM_MUSE)
V_MUSE = np.ma.masked_invalid(V_MUSE)

print('Detecting HII regions')

if(MUSE_1sig == -1.0):
    
    MUSE_1sig = np.nanmean(eF_Ha_MUSE)
    print(MUSE_1sig)

if(MUSE_1sig_V == -1.0):
    
    eV_MUSE = gaussian_filter(eF_Ha_MUSE, sigma=FWHM_MUSE)
    mean_MUSE_1sig_V = np.nanmean(eV_MUSE)
    median_MUSE_1sig_V = np.nanmedian(eV_MUSE)
    MUSE_1sig_V = np.min(np.array([mean_MUSE_1sig_V, median_MUSE_1sig_V]))

    if(np.isnan(MUSE_1sig_V)):
        MUSE_1sig_V = MUSE_1sig

blobs_final,blobs_F_Ha,image_HII,diff_map_final,diff_points,diff_Flux=HIIblob(F_Ha_MUSE,V_MUSE,FWHM_MUSE, MUSE_1sig=MUSE_1sig, MUSE_1sig_V=MUSE_1sig_V, plot=plot, refined=refined, name=name)

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

file_table_HII = DIR+'/'+name+'.HIIblob_HII.tab.hdf5'
table_HII.write(file_table_HII, overwrite=True)  
  

table_DIG = Table(dict_DIG)
for key in dict_DIG.keys():
    table_DIG[key].unit=dict_DIG_units[key]

file_table_DIG = DIR+'/'+name+'.HIIblob_DIG.tab.hdf5'
table_DIG.write(file_table_DIG, overwrite=True)  
  

hdu_HII = fits.PrimaryHDU(data = image_HII)
hdu_HII.writeto(DIR+'/'+name+"_image_HII.fits", output_verify="ignore", overwrite=True)

hdu_DIG = fits.PrimaryHDU(data = diff_map_final)
hdu_DIG.writeto(DIR+'/'+name+"_image_DIG.fits", output_verify="ignore",overwrite=True)

if (def_DIG==1):
    Ha_image_clean = F_Ha_MUSE - image_HII
    diff_map_final =  create_diff_new(Ha_image_clean,blobs_final,FWHM_MUSE,diff_points)
    hdu_DIG = fits.PrimaryHDU(data = diff_map_final)
    hdu_DIG.writeto(DIR+'/'+name+"_image_DIG.fits", output_verify="ignore",overwrite=True)

if (maps_seg==1):
    
    print('Create segmentation maps')
    map_seg=create_HII_seg(blobs_final, nx, ny)
    map_mask_seg=create_mask_HII_seg(blobs_final, nx, ny)
    
    print('Saving and writing output files')
    hdu_map_seg = fits.PrimaryHDU(data = map_seg)
    hdu_map_seg.header['OBJECT']=name
    hdu_map_seg.writeto(DIR+'/'+name+"_map_seg.fits", output_verify="ignore", overwrite=True)
    
    hdu_map_mask_seg = fits.PrimaryHDU(data = map_mask_seg)
    hdu_map_mask_seg.header['OBJECT']=name
    hdu_map_mask_seg.writeto(DIR+'/'+name+"_map_mask_seg.fits", output_verify="ignore", overwrite=True)    


print('Extracting flux_elines')

WCS, hdu_HII, hdu_DIG, table_HII, table_DIG = extracting_flux_elines(name, data, hdr_fe, 
                                                                            blobs_final, diff_points,FWHM_MUSE, plot=plot)

if (def_DIG==1):
  
    print('Extracting new DIG')
    
    #data_DIG = hdu_DIG.data
    #hdr_fe_DIG = hdu_DIG.header

    #data_HII = hdu_HII.data
    #hdr_fe_HII = hdu_HII.header
    
    cube_clean = data-hdu_HII.data
    
    cube_diff = create_diff_cube(cube_clean,blobs_final,FWHM_MUSE,diff_points)
    nz = hdr_fe['NAXIS3']
    nz_flux = int(nz/8)
    data_DIG = hdu_DIG.data 
    data_DIG[:nz_flux,:,:] = cube_diff[:nz_flux,:,:]
    hdu_HII.data = None
    hdu_DIG.data = None

    WCS, hdu_HII, hdu_DIG, table_HII, table_DIG = extracting_flux_elines(name, data, hdr_fe, blobs_final, diff_points, FWHM_MUSE, plot=plot, def_DIG=def_DIG,cube_DIG=data_DIG)
    
    #hdu_DIG.data = data_DIG
    nz_min_flux = 0 #
    nz_max_flux = nz_flux-1 #
    nz_min_EW = 3*nz_flux #
    nz_max_EW = 4*nz_flux-1 #
    # EW
    data_DIG[nz_min_EW:nz_max_EW,:,:] = (data_DIG[nz_min_flux:nz_max_flux,:,:]/data[nz_min_flux:nz_max_flux,:,:])*data[nz_min_EW:nz_max_EW,:,:]#
    hdu_DIG.data = data_DIG #
    data_DIG = None

hdu.close()    

print('Saving outputsfile flux_elines')

file_HII = DIR+"/HII."+name+".flux_elines.cube.fits.gz"
file_DIG = DIR+"/DIG."+name+".flux_elines.cube.fits.gz"

file_table_HII = DIR+"/HII."+name+".flux_elines.table.hdf5"
file_table_DIG = DIR+"/DIG."+name+".flux_elines.table.hdf5"

hdu_HII.writeto(file_HII,  overwrite=True)
hdu_DIG.writeto(file_DIG, overwrite=True)

table_HII.write(file_table_HII, overwrite=True)
table_DIG.write(file_table_DIG, overwrite=True)

print('Extracting SSP')

hdu_SSP_HII, hdu_SSP_DIG, table_SSP_HII, table_SSP_DIG = extracting_ssp(name, data_SSP, hdr_ssp, WCS, 
                                                                                    blobs_final, diff_points, FWHM_MUSE, plot=plot)

print('Saving outputsfile SSP')

file_SSP_HII = DIR+"/HII."+name+'.SSP.cube.fits.gz'
file_SSP_DIG = DIR+"/DIG."+name+'.SSP.cube.fits.gz'

hdu_SSP_HII.writeto(file_SSP_HII, overwrite=True)
hdu_SSP_DIG.writeto(file_SSP_DIG, overwrite=True)

file_table_SSP_HII = DIR+"/HII."+name+'.SSP.table.hdf5'
file_table_SSP_DIG = DIR+"/DIG."+name+'.SSP.table.hdf5'

table_SSP_HII.write(file_table_SSP_HII, overwrite=True)
table_SSP_DIG.write(file_table_SSP_DIG, overwrite=True)

print('Extracting SFH')

hdu_SFH_HII, hdu_SFH_DIG, table_SFH_HII, table_SFH_DIG = extracting_sfh(name, data_SFH, hdr_sfh, WCS, 
                                                                                    blobs_final, diff_points, FWHM_MUSE, plot=plot)

print('Saving outputsfile SFH')

file_SFH_HII = DIR+"/HII."+name+'.SFH.cube.fits.gz'
file_SFH_DIG = DIR+"/DIG."+name+'.SFH.cube.fits.gz'

hdu_SFH_HII.writeto(file_SFH_HII, overwrite=True)
hdu_SFH_DIG.writeto(file_SFH_DIG, overwrite=True)

file_table_SFH_HII = DIR+"/HII."+name+'.SFH.table.hdf5'
file_table_SFH_DIG = DIR+"/DIG."+name+'.SFH.table.hdf5'

table_SFH_HII.write(file_table_SFH_HII, overwrite=True)
table_SFH_DIG.write(file_table_SFH_DIG, overwrite=True)

print('Extracting INDEX')

hdu_INDEX_HII, hdu_INDEX_DIG, table_INDEX_HII, table_INDEX_DIG = extracting_index(name, data_INDEX, hdr_index, WCS, 
                                                                                    blobs_final, diff_points, FWHM_MUSE, plot=plot)

print('Saving outputsfile INDEX')

file_INDEX_HII = DIR+"/HII."+name+'.INDEX.cube.fits.gz'
file_INDEX_DIG = DIR+"/DIG."+name+'.INDEX.cube.fits.gz'

hdu_INDEX_HII.writeto(file_INDEX_HII, output_verify="ignore", overwrite=True)
hdu_INDEX_DIG.writeto(file_INDEX_DIG, output_verify="ignore", overwrite=True)

file_table_INDEX_HII = DIR+"/HII."+name+'.INDEX.table.hdf5'
file_table_INDEX_DIG = DIR+"/DIG."+name+'.INDEX.table.hdf5'

table_INDEX_HII.write(file_table_INDEX_HII, overwrite=True)
table_INDEX_DIG.write(file_table_INDEX_DIG, overwrite=True)

print('End program')
