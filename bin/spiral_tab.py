#!/usr/bin/python3
import numpy as np
import scipy as sp
from astropy.io import fits
from astropy.table import QTable, Table, Column
import matplotlib.pyplot as plt
import PIL
from PIL import Image,ImageEnhance
import matplotlib.colors as colors
from math import sqrt
import sys
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter, correlate
from scipy.spatial import Delaunay, cKDTree, KDTree
from scipy.interpolate import griddata
from astropy.io import ascii
from scipy.optimize import minimize, curve_fit, leastsq
import seaborn as sns
import warnings
import pandas as pd
from astropy.table import QTable, Table, Column
from astropy import units as u
import matplotlib.image as mpimg
from pyHIIExplorer.HIIblob import *
from pyHIIExplorer.extract import *
from create_spiral import *
import os
import time
import argparse
from matplotlib import rcParams as rc
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser(description='###Program to create spiral galaxy###', usage='spiral_tab.py name r PA ab A B C N_sp x_c y_c size FWHM spax_scale cont_peak f_leak plot DIR [--OPTIONAL_ARGUMENTS=*]\nRun with -h for details on the inputs\n ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('name',default='galaxy',type=str, help='Name of the galaxy')
parser.add_argument('r', type=float, help='Radius of the galaxy')
parser.add_argument('PA', type=float, help='Pitch angle')
parser.add_argument('ab', type=float, help='Half-axis ratio')
parser.add_argument('A', type=float, help='Scale parameter')
parser.add_argument('B', type=float, help='Controls the “bar/bulge-to-arm” size')
parser.add_argument('C', type=float, help='Controls the tightness much like the Hubble scheme')
parser.add_argument('N_sp', type=int, help='Number of arms')
parser.add_argument('x_c', type=int, help='Center on x axis (spaxels)')
parser.add_argument('y_c', type=int, help='Center on y axis (spaxels)')
parser.add_argument('size', type=int, help='Size of galaxy (spaxels)')
parser.add_argument('FWHM', type=float, help='FWHM')
parser.add_argument('spax_scale', type=float, help='Spax_scale')
parser.add_argument('cont_peak', type=float, help='Cont_peak')
parser.add_argument('f_leak', type=float, help='leaking')
parser.add_argument('plot', type=int, help='Save all plots in DIR')
parser.add_argument('DIR', type=str, help='Where save outputfiles')

args = parser.parse_args()

print('Reading files')

name = args.name
r = args.r
PA = args.PA
ab = args.ab
A = args.A
B = args.B
C = args.C
N_sp = args.N_sp
x_c = args.x_c
y_c = args.y_c
size = args.size
FWHM = args.FWHM
spax_scale = args.spax_scale
cont_peak = args.cont_peak
f_leak = args.f_leak
plot = args.plot
DIR = args.DIR

plt.ion()


rc.update({'font.size': 24,\
           'font.weight': 1000,\
           'text.usetex': True,\
           'path.simplify'           :   True,\
           'xtick.labelsize' : 22,\
           'ytick.labelsize' : 22,\
#           'xtick.major.size' : 3.5,\
#           'ytick.major.size' : 3.5,\
           'axes.linewidth'  : 2.0,\
               # Increase the tick-mark lengths (defaults are 4 and 2)
           'xtick.major.size'        :   6,\
           'ytick.major.size'        :   6,\
           'xtick.minor.size'        :   3,\
           'ytick.minor.size'        :   3,\
           'xtick.major.width'       :   1,\
           'ytick.major.width'       :   1,\
           'lines.markeredgewidth'   :   1,\
           'legend.numpoints'        :   1,\
           'xtick.minor.width'       :   1,\
           'ytick.minor.width'       :   1,\
           'legend.frameon'          :   False,\
           'legend.handletextpad'    :   0.3,\
           'font.family'    :   'serif',\
           'mathtext.fontset'        :   'stix',\
           'axes.facecolor' : "w",\
          })

#####Create spiral####
x_sp,y_sp,r_sp,x_sp_r,y_sp_r,N2_sp,O3_sp,EW_sp,r_dist = create_spiral(r=r,ab=ab,PA=PA, A=A,B=B,C=C,N_sp=N_sp)

r_dist=size*r_dist
x_e_sp=x_c+size*x_sp
y_e_sp=y_c+size*y_sp

x_e_sp_r=x_c+size*x_sp_r
y_e_sp_r=y_c+size*y_sp_r

nx_in = 2*x_c
ny_in = 2*y_c

h_scale = 10/spax_scale
r_disk = r_dist/h_scale
cont = cont_peak*np.exp(-r_disk)
rat_NII_Ha = 10**(N2_sp)
Ha = 10**(EW_sp) * cont 
NII = rat_NII_Ha * Ha
Hb = Ha/2.86 
OIII = Hb*10**(O3_sp)
log_U = -1.40*np.log10(rat_NII_Ha*2.86)-3.26
rad_HII_arc = 0.032219499585429647 * np.sqrt(Ha) * 10 ** ((-log_U-2.8)/2) #* (1/0.8) # kpc->arcsec
rad_HII_arc = rad_HII_arc / spax_scale
R_in = rad_HII_arc/np.sqrt(2)
X_in = x_e_sp_r
Y_in = y_e_sp_r
flux_in  = Ha
blobs_in = np.swapaxes(np.array((Y_in,X_in,R_in)),axis1=0,axis2=1)

####Create cont####
#img_cont = create_cont(ny_in=ny_in, nx_in=nx_in, h_scale=h_scale, cont_peak=cont_peak, x_c=x_c, y_c=y_c, PA=PA, ab=ab)/(np.mean(r_sp)*2.5)**2
#/(spax_scale/FWHM)**2
img_cont = create_cont(cont, y_sp, x_sp)
####Create leaking####
img_Ha_leak = create_HII_image_leaking(blobs_in,Ha,nx_in,ny_in,dr=30, fleak=f_leak)
img_Hb_leak = create_HII_image_leaking(blobs_in,Hb,nx_in,ny_in,dr=30, fleak=f_leak)
img_NII_leak = create_HII_image_leaking(blobs_in,NII,nx_in,ny_in,dr=30, fleak=f_leak*1.5)
img_OIII_leak = create_HII_image_leaking(blobs_in,OIII,nx_in,ny_in,dr=30, fleak=f_leak*1.2)
ew_Ha_leak = img_Ha_leak/img_cont

cube_leak=np.array((img_Hb_leak,img_OIII_leak,img_Ha_leak,img_NII_leak, ew_Ha_leak))

hdu_hdr = {}
hdu_hdr['CRVAL1'] = nx_in/2*spax_scale
hdu_hdr['CRVAL2'] = ny_in/2*spax_scale
hdu_hdr['CRPIX1'] = nx_in/2
hdu_hdr['CRPIX2'] = ny_in/2
hdu_hdr['CDELT1'] = (-1)*spax_scale
hdu_hdr['CDELT2'] = spax_scale
hdu_hdr['NAME0'] = 'flux_Hb_leak'
hdu_hdr['NAME1'] = 'flux_[OIII]_leak'
hdu_hdr['NAME2'] = 'flux_Ha_leak'
hdu_hdr['NAME3'] = 'flux_[NII]_leak'
hdu_hdr['NAME4'] = 'EW_Ha_leak'

hdu_hdr = fits.Header(hdu_hdr)
hdu_cube_leak = fits.PrimaryHDU(data = cube_leak, header=hdu_hdr)
hdu_cube_leak.writeto(DIR+'/'+'leak.cube.fits.gz', overwrite=True)

if (plot==1):
    plt.imshow(img_Ha_leak)
    plt.xlim(0,nx_in)
    plt.ylim(0,ny_in)
    plt.savefig(DIR+'/'+name+'_leak.png')
    plt.show()

####Create holmes####
cube_holmes = create_DIG (x_c, y_c , PA, ab, h_scale, cont_peak, size)


print('DIG1:')
print(np.mean(cube_holmes[4,:,:]), np.std(cube_holmes[4,:,:]), np.min(cube_holmes[4,:,:]), np.max(cube_holmes[4,:,:]))

f_scale = 1
hdu_cube_holmes = fits.PrimaryHDU(data = cube_holmes*f_scale)
hdu_cube_holmes.writeto(DIR+'/'+'holmes.cube.fits.gz', overwrite=True)

if (plot==1):
    plt.imshow(cube_holmes[2,:,:])
    plt.xlim(0,nx_in)
    plt.ylim(0,ny_in)
    plt.savefig(DIR+'/'+name+'_holmes.png')
    plt.show()

####Create HII regions####
img_Ha = create_HII_image(blobs_in,Ha,nx_in,ny_in,dr=3)
img_Hb = create_HII_image(blobs_in,Hb,nx_in,ny_in,dr=3)
img_NII = create_HII_image(blobs_in,NII,nx_in,ny_in,dr=3)
img_OIII = create_HII_image(blobs_in,OIII,nx_in,ny_in,dr=3)
ew_Ha = img_Ha/img_cont

img_cont_Ha = gaussian_filter((img_Ha*2), sigma=2)


cube_HII=np.array((img_Hb,img_OIII,img_Ha,img_NII, ew_Ha))

hdu_hdr = {}
hdu_hdr['CRVAL1'] = nx_in/2*spax_scale
hdu_hdr['CRVAL2'] = ny_in/2*spax_scale
hdu_hdr['CRPIX1'] = nx_in/2
hdu_hdr['CRPIX2'] = ny_in/2
hdu_hdr['CDELT1'] = (-1)*spax_scale
hdu_hdr['CDELT2'] = spax_scale
hdu_hdr['NAME0'] = 'flux_Hb'
hdu_hdr['NAME1'] = 'flux_[OIII]'
hdu_hdr['NAME2'] = 'flux_Ha'
hdu_hdr['NAME3'] = 'flux_[NII]'
hdu_hdr['NAME4'] = 'EW_Ha'

hdu_hdr = fits.Header(hdu_hdr)
hdu_cube_HII = fits.PrimaryHDU(data = cube_HII*f_scale, header=hdu_hdr)
hdu_cube_HII.writeto(DIR+'/'+'HIIregions.cube.fits.gz', overwrite=True)

if (plot==1):
    plt.imshow(img_Ha)
    plt.xlim(0,nx_in)
    plt.ylim(0,ny_in)
    plt.savefig(DIR+'/'+name+'_onlyHIIregions.png')
    plt.show()

####Create cube no leak####
cube_noleak = cube_HII + cube_holmes

hdu_hdr = fits.Header(hdu_hdr)
hdu_cube_noleak = fits.PrimaryHDU(data = cube_noleak*f_scale, header=hdu_hdr)
hdu_cube_noleak.writeto(DIR+'/'+'sim_noleak.cube.fits.gz', overwrite=True)

####Create cube holmes+leaking####
hdu_DIG_all = fits.PrimaryHDU(data = (cube_holmes*f_scale)+cube_leak, header=hdu_hdr)
hdu_DIG_all.writeto(DIR+'/'+'diff_all.cube.fits.gz', overwrite=True)



####Create cube all: holmes, leaking, HIIregions####
cube_all = (cube_HII + cube_holmes)*f_scale
#cube_all[4,:,:] = cube_all[4,:,:] *f_scale

cube_ALL = cube_all+cube_leak

#cube_ALL = cube_HII*f_scale

hdu_hdr_ALL = fits.Header(hdu_hdr)
hdu_cube_ALL = fits.PrimaryHDU(data = cube_ALL, header=hdu_hdr) 
hdu_cube_ALL.writeto(DIR+'/'+'sim.cube.fits.gz', overwrite=True)


if (plot==1):
    os.system('/usr/bin/python3 RGB_img_cube_sim.py datos/sim.cube.fits.gz sim_test.png') 
    img_all = mpimg.imread('sim_test.png')
    plt.imshow(img_all)
    plt.xlim(0,nx_in)
    plt.ylim(0,ny_in)
    plt.savefig(DIR+'/'+name+'_all_RGB.png')

img_Ha_ALL = cube_ALL[2,:,:]


FWHM=FWHM/spax_scale 

blobs_final,blobs_F_Ha,image_HII, diff_map_final,diff_points,diff_Flux = HIIblob(img_Ha_ALL,img_cont_Ha,FWHM, MUSE_1sig=1.0, MUSE_1sig_V=0.1,
                                                                                 plot=1, refined=3, num_sigma=300, max_size=2, name=name,
                                                                                 DIR=DIR)

WCS, hdu_HII_out, hdu_DIG_out, table_HII, table_DIG = extracting_flux_elines('sim', hdu_cube_ALL, blobs_final, diff_points, FWHM,  plot=0)

cube_diff_output = create_diff_cube(hdu_cube_ALL.data-hdu_HII_out.data,blobs_final,FWHM,diff_points)

hdu_DIG_out.data = cube_diff_output

hdu_SUM_out = hdu_HII_out.copy()
hdu_RES_out = hdu_HII_out.copy()
hdu_SUM_out.data = hdu_SUM_out.data+hdu_DIG_out.data
hdu_RES_out.data = hdu_cube_ALL.data-(hdu_SUM_out.data)


hdu_HII_out.writeto(DIR+'/'+'HII_out.cube.fits.gz', overwrite=True)
hdu_DIG_out.writeto(DIR+'/'+'DIG_out.cube.fits.gz', overwrite=True)
hdu_SUM_out.writeto(DIR+'/'+'SUM_out.cube.fits.gz', overwrite=True)
hdu_RES_out.writeto(DIR+'/'+'RES_out.cube.fits.gz', overwrite=True)


os.system('/usr/bin/python3 RGB_img_cube_sim.py datos/HII_out.cube.fits.gz datos/HII_out.png')
os.system('/usr/bin/python3 RGB_img_cube_sim.py datos/HIIregions.cube.fits.gz datos/HII_inp.png')

os.system('/usr/bin/python3 RGB_img_cube_sim.py datos/DIG_out.cube.fits.gz datos/DIG_out.png')
os.system('/usr/bin/python3 RGB_img_cube_sim.py datos/diff_all.cube.fits.gz datos/DIG_inp.png')

os.system('/usr/bin/python3 RGB_img_cube_sim.py datos/SUM_out.cube.fits.gz datos/SUM_out.png')
os.system('/usr/bin/python3 RGB_img_cube_sim.py datos/sim.cube.fits.gz datos/SIM_inp.png')


##################################

O3_HII_inp = O3_sp
N2_HII_inp = N2_sp
ew_HII_inp = EW_sp

###################################

Ha_HII_out = table_HII['flux_Ha']
Hb_HII_out = table_HII['flux_Hb']
OIII_HII_out = table_HII['flux_[OIII]']
NII_HII_out = table_HII['flux_[NII]']
ew_HII_out = table_HII['EW_Ha']

O3_HII_out = np.array((np.log10(OIII_HII_out)-np.log10(Hb_HII_out)).flat)
N2_HII_out = np.array((np.log10(NII_HII_out)-np.log10(Ha_HII_out)).flat)
ew_HII_out = np.array((np.log10(ew_HII_out)).flat)

##################################

cube_DIG_inp = hdu_DIG_all.data
Ha_DIG_inp = cube_DIG_inp[2,:,:]
Hb_DIG_inp = cube_DIG_inp[0,:,:] 
OIII_DIG_inp = cube_DIG_inp[1,:,:]
NII_DIG_inp = cube_DIG_inp[3,:,:]
ew_DIG_inp = cube_DIG_inp[4,:,:]

O3_DIG_inp = np.array((np.log10(OIII_DIG_inp)-np.log10(Hb_DIG_inp)).flat)
N2_DIG_inp = np.array((np.log10(NII_DIG_inp)-np.log10(Ha_DIG_inp)).flat)
ew_DIG_inp = np.array(np.log10(ew_DIG_inp).flat)

#################################

cube_DIG_out = hdu_DIG_out.data
Ha_DIG_out = cube_DIG_out[2,:,:]
Hb_DIG_out = cube_DIG_out[0,:,:] 
OIII_DIG_out = cube_DIG_out[1,:,:]
NII_DIG_out = cube_DIG_out[3,:,:]
ew_DIG_out = cube_DIG_out[4,:,:]


O3_DIG_out = np.array((np.log10(OIII_DIG_out)-np.log10(Hb_DIG_out)).flat)
N2_DIG_out = np.array((np.log10(NII_DIG_out)-np.log10(Ha_DIG_out)).flat)
ew_DIG_out = np.array(np.log10(ew_DIG_out).flat)

#################################
ax = plt.figure(figsize=(5,5))

df1 = pd.DataFrame({'NII Halfa':N2_HII_inp, 'W':ew_HII_inp})
df2 = pd.DataFrame({'NII Halfa2':N2_HII_out, 'W2': ew_HII_out})

ax=sns.regplot(data=df1, x='NII Halfa', y='W',fit_reg = False, scatter_kws={'alpha':1, 'color':'#7bffa7', 'edgecolor':'none'})
sns.regplot(data=df2, x='NII Halfa2', y='W2', ax=ax, fit_reg = False, scatter_kws={'alpha':1,'color':'#008f39', 'edgecolor':'none'})

plt.xlabel(r'log([NII]/H$\alpha$)', size=33)
plt.ylabel(r'log W$_{H\alpha}$ ($\AA$)', size=33)
plt.axvline(-0.4, 0.24, c='#646464', ls="--", linewidth=2)
plt.axhline(0.79, 0.37, 1, c='#646464', ls="--", linewidth=2)
plt.axhline(0.5, c='#646464', ls="--", linewidth=2)

plt.text(-1, 3.5, 'SF', fontsize=35, fontweight=800)
plt.text(0.3, 3.5, 'sAGN', fontsize=35, fontweight=800)
plt.text(0.25, 0.5, 'wAGN', fontsize=35, fontweight=800)
plt.text(-1, -0.5, 'Retired', fontsize=35, fontweight=800)

plt.xlim(-1.3, 1.1)
plt.ylim(-1.3, 4)
plt.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.2, wspace=0, hspace=0)
plt.savefig(DIR+'/'+name+'__WHAN_HII.png')
plt.show()

########################

#######################
ax = plt.figure(figsize=(5,5))

df1 = pd.DataFrame({'NII Halfa': N2_DIG_inp, 'W': ew_DIG_inp})
df2 = pd.DataFrame({'NII Halfa2':N2_DIG_out, 'W2': ew_DIG_out})

print('DIG2:')
print(np.mean(ew_DIG_inp), np.std(ew_DIG_inp), np.min(ew_DIG_inp), np.max(ew_DIG_inp))


ax=sns.regplot(data=df1, x='NII Halfa', y='W',fit_reg = False, scatter_kws={'alpha':0.4, 'color':'#ff7214', 'edgecolor':'none'})
sns.regplot(data=df2, x='NII Halfa2', y='W2', ax=ax, fit_reg = False, scatter_kws={'alpha':0.4,'color':'#a80000', 'edgecolor':'none'})

plt.xlabel(r'log([NII]/H$\alpha$)', size=33)
plt.ylabel(r'log W$_{H\alpha}$ ($\AA$)', size=33)
plt.axvline(-0.4, 0.24, 1, c='#646464', ls="--", linewidth=2)
plt.axhline(0.79, 0.37, 1, c='#646464', ls="--", linewidth=2)
plt.axhline(0.5, c='#646464', ls="--", linewidth=2)

plt.text(-1,3.5 , 'SF', fontsize=35, fontweight=800)
plt.text(0.3, 3.5, 'sAGN', fontsize=35, fontweight=800)
plt.text(0.25, 0.5, 'wAGN', fontsize=35, fontweight=800)
plt.text(-1, -0.5, 'Retired', fontsize=35, fontweight=800)

plt.xlim(-1.3, 1.1)
plt.ylim(-1.3, 4)
plt.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.2, wspace=0, hspace=0)
plt.savefig(DIR+'/'+name+'__WHAN_DIG.png')
plt.show()



####################################################################################

x_min1=-4
x_max1=0.45
y_min1=-2.5
y_max1=1.2

bpt_hiiregions = create_bpt_sns_double(x_plt1=N2_HII_inp, y_plt1=O3_HII_inp, x_plt2=N2_HII_out, y_plt2=O3_HII_out, cmap1='winter_r', cmap2='PiYG_r',
                                       c1='#7bffa7', c2='#008f39', x_min1=x_min1, x_max1=x_max1, 
                                       y_min1=y_min1,y_max1=y_max1, legend=['Input HII', 'Output HII'],
                                       namesave=DIR+'/'+name+'bpt_hiiregions')


bpt_dig = create_bpt_sns_double(x_plt1=N2_DIG_inp, y_plt1=O3_DIG_inp, x_plt2=N2_DIG_out, y_plt2=O3_DIG_out, cmap1='autumn',
                                cmap2='seismic_r', c1='#ff7214', c2='#a80000', x_min1=x_min1, x_max1=x_max1, 
                                y_min1=y_min1,y_max1=y_max1, legend=['Input DIG', 'Output DIG'],
                                namesave=DIR+'/'+name+'bpt_digregions')

img_HII_out = mpimg.imread(DIR+'/'+'HII_out.png')
img_HII_inp = mpimg.imread(DIR+'/'+'HII_inp.png')

img_DIG_out = mpimg.imread(DIR+'/'+'DIG_out.png')
img_DIG_inp = mpimg.imread(DIR+'/'+'DIG_inp.png')

img_SUM = mpimg.imread(DIR+'/'+'SUM_out.png')
img_SIM_inp = mpimg.imread(DIR+'/'+'SIM_inp.png')

img_bpt_hiiregions = mpimg.imread(DIR+'/'+name+'bpt_hiiregions.png')
img_bpt_dig = mpimg.imread(DIR+'/'+name+'bpt_digregions.png')

img_whan_hiiregions = mpimg.imread(DIR+'/'+name+'__WHAN_HII.png')
img_whan_dig = mpimg.imread(DIR+'/'+name+'__WHAN_DIG.png')


fig, ax = plt.subplots(2,5,figsize=(45,18))

plt.setp(ax, xticks=([]),yticks=([]))

ax[0][0].imshow(img_SIM_inp, interpolation='none')
ax[1][0].imshow(img_SUM, interpolation='none')

ax[0][1].imshow(img_HII_inp, interpolation='none')
ax[1][1].imshow(img_HII_out, interpolation='none')

ax[0][2].imshow(img_DIG_inp, interpolation='none')
ax[1][2].imshow(img_DIG_out, interpolation='none')

ax[0][3].imshow(img_bpt_hiiregions, interpolation='none')
ax[1][3].imshow(img_bpt_dig, interpolation='none')

ax[0][4].imshow(img_whan_hiiregions, interpolation='none')
ax[1][4].imshow(img_whan_dig, interpolation='none')


ax[0][3].axis('off')
ax[1][3].axis('off')
ax[0][4].axis('off')
ax[1][4].axis('off')


plt.subplots_adjust(left = 0.02, right=1, bottom = 0.02, top=0.99, wspace = 0.006,  hspace = 0.006)

plt.savefig(DIR+'/'+name+'all.pdf')


##################################

cube_HII_inp = hdu_cube_HII.data
Ha_HII_inp = cube_HII_inp[2,:,:]
Hb_HII_inp = cube_HII_inp[0,:,:] 
OIII_HII_inp = cube_HII_inp[1,:,:]
NII_HII_inp = cube_HII_inp[3,:,:]
ew_HII_inp = cube_HII_inp[4,:,:]

O3_HII_inp = np.array((np.log10(OIII_HII_inp)-np.log10(Hb_HII_inp)).flat)
N2_HII_inp = np.array((np.log10(NII_HII_inp)-np.log10(Ha_HII_inp)).flat)
ew_HII_inp = np.array(ew_HII_inp.flat)
Ha_HII_inp = np.array(Ha_HII_inp.flat)

cube_HII_out = hdu_HII_out.data
Ha_HII_out = cube_HII_out[2,:,:]
Hb_HII_out = cube_HII_out[0,:,:] 
OIII_HII_out = cube_HII_out[1,:,:]
NII_HII_out = cube_HII_out[3,:,:]
ew_HII_out = cube_HII_out[4,:,:]

O3_HII_out = np.array((np.log10(OIII_HII_out)-np.log10(Hb_HII_out)).flat)
N2_HII_out = np.array((np.log10(NII_HII_out)-np.log10(Ha_HII_out)).flat)
ew_HII_out = np.array(ew_HII_out.flat)
Ha_HII_out = np.array(Ha_HII_out.flat)

##################################

ew_DIG_inp = np.array((cube_DIG_inp[4,:,:]).flat)
Ha_DIG_inp = np.array(cube_DIG_inp[2,:,:].flat)


ew_DIG_out = np.array((cube_DIG_out[4,:,:]).flat)
Ha_DIG_out = np.array((cube_DIG_out[2,:,:]).flat)


##################################


mask_flux_05 = (blobs_final>0.5*np.median(flux_in))
mask_flux_05_in = (blobs_in>0.5*np.median(flux_in))

mask_flux_rad05 = (flux_in>0.5*np.median(flux_in))
mask_flux_bl05 = (blobs_F_Ha>0.5*np.median(flux_in))


mask_N2_HII = (N2_HII_inp>-100000000)&(N2_HII_inp<100000000)&(N2_HII_out>-100000000)&(N2_HII_out<100000000)
mask_O3_HII = (O3_HII_inp>-100000000)&(O3_HII_inp<100000000)&(O3_HII_out>-100000000)&(O3_HII_out<100000000)

mask_N2_DIG = (N2_DIG_inp>-100000000)&(N2_DIG_inp<100000000)&(N2_DIG_out>-100000000)&(N2_DIG_out<100000000)
mask_O3_DIG = (O3_DIG_inp>-100000000)&(O3_DIG_inp<100000000)&(O3_DIG_out>-100000000)&(O3_DIG_out<100000000)

mask_ew_HII = (ew_HII_inp>0.5*np.median(ew_HII_inp))
mask_ew_DIG = (ew_DIG_inp>0.5*np.median(ew_DIG_inp))

table_info=Table(rows=[( N_sp, ab*100, len(blobs_in), len(blobs_final), np.around(np.float((100*len(blobs_final))/len(blobs_in)), 3), np.around(np.float((100*len(blobs_final[mask_flux_05]))/len(blobs_in[mask_flux_05_in])),3), np.around(np.float(np.mean(((Ha_HII_out-Ha_HII_inp)/Ha_HII_inp)[Ha_HII_inp>0.5*np.median(Ha_HII_inp)])),3), np.around(np.float(np.std(((Ha_HII_out-Ha_HII_inp)/Ha_HII_inp)[Ha_HII_inp>0.5*np.median(Ha_HII_inp)])),3),np.around(np.float(np.mean(R_in[mask_flux_rad05])),3), np.around(np.float(np.std(R_in[mask_flux_rad05])),3), np.around(np.float(np.mean(blobs_final[:,2][mask_flux_bl05])),3), np.around(np.float(np.std(blobs_final[:,2][mask_flux_bl05])),3), np.around(np.float(np.mean(N2_HII_out[mask_N2_HII]-N2_HII_inp[mask_N2_HII])),3), np.around(np.float(np.std(N2_HII_out[mask_N2_HII]-N2_HII_inp[mask_N2_HII])),3), np.around(np.float(np.mean(O3_HII_out[mask_O3_HII]-O3_HII_inp[mask_O3_HII])),3), np.around(np.float(np.std(O3_HII_out[mask_O3_HII]-O3_HII_inp[mask_O3_HII])),3), np.around(np.float(np.mean(((ew_HII_out-ew_HII_inp)/ew_HII_inp)[mask_ew_HII])),3), np.around(np.float(np.std(((ew_HII_out-ew_HII_inp)/ew_HII_inp)[mask_ew_HII])),3), np.around(np.float(np.mean(N2_DIG_out[mask_N2_DIG]-N2_DIG_inp[mask_N2_DIG])),3), np.around(np.float(np.std(N2_DIG_out[mask_N2_DIG]-N2_DIG_inp[mask_N2_DIG])),3), np.around(np.float(np.mean(O3_DIG_out[mask_O3_DIG]-O3_DIG_inp[mask_O3_DIG])),3), np.around(np.float(np.std(O3_DIG_out[mask_O3_DIG]-O3_DIG_inp[mask_O3_DIG])),3), np.around(np.float(np.mean(((ew_DIG_out-ew_DIG_inp)/ew_DIG_inp)[mask_ew_DIG])),3), np.around(np.float(np.std(((ew_DIG_out-ew_DIG_inp)/ew_DIG_inp)[mask_ew_DIG])),3)  )])




file_table_info = DIR+'/info_'+name+'.table.ecsv'
ascii.write(table_info, file_table_info, format='no_header', overwrite=True, delimiter='&')

print('end')

