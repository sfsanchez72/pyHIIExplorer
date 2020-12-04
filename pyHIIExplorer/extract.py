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
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from scipy.optimize import minimize
from astropy.table import QTable, Table, Column
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from pyHIIExplorer.HIIblob import *
import warnings
warnings.filterwarnings('ignore')


def create_HII_seg(blobs, nx, ny):
    image = np.zeros((ny,nx))
    indx = 1
    for blob in blobs:
        y, x, r = blob
        x_g = np.arange(0, nx, 1)
        y_g = np.arange(0, ny, 1)
        x_g, y_g = np.meshgrid(x_g, y_g)
        dist = np.sqrt((x_g-x)**2+(y_g-y)**2)
        image[dist<=2.5*r]=indx
        indx = indx+1
    return image


def create_mask_HII_seg(blobs, nx, ny):
    image = np.ones((ny,nx))
    for blob in blobs:
        y, x, r = blob
        x_g = np.arange(0, nx, 1)
        y_g = np.arange(0, ny, 1)
        x_g, y_g = np.meshgrid(x_g, y_g)
        dist = np.sqrt((x_g-x)**2+(y_g-y)**2)
        image[dist<=2.5*r]=0
    return image


def extracting_flux_elines(name, hdu, blobs_final, diff_points, plot=0, def_DIG=0, cube_DIG=np.array):

#def extracting_flux_elines(name, hdu, blobs_final, diff_points, plot=0, def_DIG=0, cube_DIG=np.array):
    
    data = hdu[0].data
    hdr_fe = hdu[0].header

    
    keys = np.array(list(hdr_fe.keys()))
    val = np.array(list(hdr_fe.values()))

    nx = hdr_fe["NAXIS1"]
    ny = hdr_fe["NAXIS2"]
    nz = hdr_fe["NAXIS3"]

    model_HII = np.zeros((nz,ny,nx))
    model_diff = np.zeros((nz,ny,nx))

    #file_HII = "HII."+name+".flux_elines.cube.fits.gz"
    #file_DIG = "DIG."+name+".flux_elines.cube.fits.gz"

    #file_table_HII = "HII."+name+".flux_elines.table.fits.gz"
    #file_table_DIG = "DIG."+name+".flux_elines.table.fits.gz"

    CRVAL1 = hdr_fe["CRVAL1"]
    CRPIX1 = hdr_fe["CRPIX1"]
    CRVAL2 = hdr_fe["CRVAL2"]
    CRPIX2 = hdr_fe["CRPIX2"]
 
    
    try:
        CDELT1 = hdr_fe["CDELT1"]
        CDELT2 = hdr_fe["CDELT2"]

    except:
        CDELT1 = hdr_fe["CD1_1"]
        CDELT2 = hdr_fe["CD2_2"]
        
    WCS = np.array((CRPIX1,CRPIX2,CRVAL1,CRVAL2,CDELT1,CDELT2))

    dict_HII = {}
    list_HII = []

    for i in range(len(blobs_final)):
        list_HII.append(name+"-"+str(i+1))
    dict_HII["HIIREGID"] = list_HII
    dict_HII["X"] = blobs_final[:,1]
    dict_HII["Y"] = blobs_final[:,0]
    dict_HII["R"] = blobs_final[:,2]
    dict_HII["RA"] = (blobs_final[:,1]-CRPIX1+1)*CDELT1+CRVAL1
    dict_HII["DEC"] = (blobs_final[:,0]-CRPIX2+1)*CDELT2+CRVAL2
    dict_HII_units = {}
    dict_HII_units["HIIREGID"] = "none"
    dict_HII_units["X"] = "spaxels"
    dict_HII_units["Y"] = "spaxels"
    dict_HII_units["R"] = "spaxels"
    dict_HII_units["RA"] = "deg"
    dict_HII_units["DEC"] = "deg"

    dict_DIG = {}
    list_DIG = []

    for i in range(len(diff_points)):
        list_DIG.append(name+"-DIG-"+str(i+1))
    dict_DIG["DIGREGID"] = list_DIG
    dict_DIG["X"] = diff_points[:,0]
    dict_DIG["Y"] = diff_points[:,1]
    dict_DIG["RA"] = (diff_points[:,0]-CRPIX1+1)*CDELT1+CRVAL1
    dict_DIG["DEC"] = (diff_points[:,1]-CRPIX2+1)*CDELT2+CRVAL2
    dict_DIG_units = {}
    dict_DIG_units["DIGREGID"] = "none"
    dict_DIG_units["X"] = "spaxels"
    dict_DIG_units["Y"] = "spaxels"
    dict_DIG_units["RA"] = "deg"
    dict_DIG_units["DEC"] = "deg"

    plot = 0

    if (plot == 1):
        fig, axes = plt.subplots(1,3,figsize=(13,5))
        fig.canvas.set_window_title("HII explorer")

    for i in range(nz):
        key_now = "NAME"+str(i)
        val_now = hdr_fe[key_now].replace(" ", "_")
        dict_HII_units[val_now] = ""
        error = 0
        kind = 1
        refined = 0
        
        if ("e_" in val_now):
            error = 1
            refined = 0
        if ("flux" in val_now):
            kind = 0
            refined = 3
            dict_HII_units[val_now] = "10^-16 erg/s/cm^2"
        if ("EW" in val_now):
            dict_HII_units[val_now] = "Angstrom"
        if ("vel" in val_now):
            dict_HII_units[val_now] = "km/s"
        if ("disp" in val_now):
            dict_HII_units[val_now] = "km/s"
        dict_DIG_units[val_now] = dict_HII_units[val_now]
        map_now = data[i,:,:]
        if ((def_DIG==1)&("flux" in val_now)&(error==0)):
            map_now = data[i,:,:] - cube_DIG[i,:,:]
            if ("flux" in val_now):
                kind = 0
                refined = 0
        if (error == 1):
            map_now = map_now**2
            
            
        flux_HII,img_HII,img_diff,diff_points,diff_val = HIIextraction(map_now,blobs_final,kind=kind, we=2, refined=refined)
        
        img_diff,diff_points,diff_Flux = create_diff(img_diff,blobs_final,FWHM_MUSE=1.0)
        
        if (error == 1):
            map_now = np.sqrt(map_now)
            flux_HII = np.sqrt(flux_HII)
            diff_Flux = np.sqrt(diff_Flux)
            img_HII = np.sqrt(img_HII)
            img_diff = np.sqrt(img_diff)

        dict_HII[val_now] = flux_HII    
        dict_DIG[val_now] = diff_Flux
        model_HII[i,:,:] = img_HII
        model_diff[i,:,:] = img_diff
    
        if (plot == 1):
            for ax in axes:
                ax.cla()
            (nx,ny) = map_now.shape
            cmap='gist_stern_r'
            im_Ha=axes[0].imshow(map_now, interpolation='none', cmap=cmap, norm=colors.PowerNorm(gamma=0.5)) 
            clim=im_Ha.properties()['clim']
            axes[1].imshow(img_HII, interpolation='none',cmap=cmap, label=r'H$\alpha$',norm=colors.PowerNorm(gamma=0.5), clim=clim) 
            axes[2].imshow(img_diff, interpolation='none',cmap=cmap, label=r'H$\alpha$',norm=colors.PowerNorm(gamma=0.5), clim=clim) 
            axes[0].set_xlim(0,nx)
            axes[0].set_ylim(0,ny)
            axes[1].set_xlim(0,nx)
            axes[1].set_ylim(0,ny)    
            axes[2].set_xlim(0,nx)
            axes[2].set_ylim(0,ny)  
            axes[0].set_title(val_now)
            fig.canvas.draw()

    hdu_HII = fits.PrimaryHDU(data = model_HII , header = hdr_fe)
    hdu_DIG = fits.PrimaryHDU(data = model_diff , header = hdr_fe)


    table_HII = Table(dict_HII)
    for key in dict_HII.keys():
        table_HII[key].unit=dict_HII_units[key]

    table_DIG = Table(dict_DIG)
    for key in dict_DIG.keys():
        table_DIG[key].unit=dict_DIG_units[key]
    
    return WCS, hdu_HII, hdu_DIG, table_HII, table_DIG

def extracting_ssp(name, hdu_ssp, WCS, blobs_final, diff_points, plot=0):
    desc_ssp = ["V", "CS", "DZ", "flux", "e_flux", "age_LW", "age_MW", "e_age", "ZH_LW", "ZH_MW", "e_ZH", "Av", "e_Av","vel_SSP", "e_vel_SSP", "disp_SSP", "e_disp_SSP", "ML", "mass", "mass_dust"]

    data_SSP=hdu_ssp[0].data
    hdr_ssp=hdu_ssp[0].header

    keys = np.array(list(hdr_ssp.keys()))
    val = np.array(list(hdr_ssp.values()))
   
    nx_ssp = hdr_ssp['NAXIS1']
    ny_ssp = hdr_ssp['NAXIS2']
    nz_ssp = hdr_ssp['NAXIS3']
   
    model_HII_SSP = np.zeros((nz_ssp,ny_ssp,nx_ssp))
    model_diff_SSP = np.zeros((nz_ssp,ny_ssp,nx_ssp))
   

    dict_HII = {}
    list_HII = []

    for i in range(len(blobs_final)):
        list_HII.append(name+"-"+str(i+1))
    dict_HII['HIIREGID'] = list_HII
    dict_HII['X'] = blobs_final[:,1]
    dict_HII['Y'] = blobs_final[:,0]
    dict_HII['R'] = blobs_final[:,2]
    dict_HII['RA'] = (blobs_final[:,1]-WCS[0]+1)*WCS[4]+WCS[2]
    dict_HII['DEC'] = (blobs_final[:,0]-WCS[1]+1)*WCS[5]+WCS[3]
    dict_HII_units = {}
    dict_HII_units['HIIREGID'] = 'none'
    dict_HII_units['X'] = 'spaxels'
    dict_HII_units['Y'] = 'spaxels'
    dict_HII_units['R'] = 'spaxels'
    dict_HII_units['RA'] = 'deg'
    dict_HII_units['DEC'] = 'deg'
   
    dict_DIG = {}
    list_DIG = []
   
    for i in range(len(diff_points)):
       list_DIG.append(name+"-DIG-"+str(i+1))
    dict_DIG['DIGREGID'] = list_DIG
    dict_DIG['X'] = diff_points[:,0]
    dict_DIG['Y'] = diff_points[:,1]
    dict_DIG['RA'] = (diff_points[:,0]-WCS[0]+1)*WCS[4]+WCS[2]
    dict_DIG['DEC'] = (diff_points[:,1]-WCS[1]+1)*WCS[5]+WCS[3]
    dict_DIG_units = {}
    dict_DIG_units['DIGREGID'] = 'none'
    dict_DIG_units['X'] = 'spaxels'
    dict_DIG_units['Y'] = 'spaxels'
    dict_DIG_units['RA'] = 'deg'
    dict_DIG_units['DEC'] = 'deg'

    plot=0
    
    if (plot == 1):
       fig, axes = plt.subplots(1,3,figsize=(13,5))
       fig.canvas.set_window_title('HII explorer')
    
    for i in range(nz_ssp):
        val_now = desc_ssp[i]
        dict_HII_units[val_now] = ''
        error = 0
        kind = 1
        refined = 0
        
        if ('e_' in val_now):
            error=1
            refined=0
        dict_DIG_units[val_now] = dict_HII_units[val_now]
  
        map_now = data_SSP[i,:,:]
        if (error == 1):
            map_now = map_now**2
        
        flux_HII,img_HII,img_diff,diff_points,diff_val = HIIextraction(map_now,blobs_final,kind=kind, we=2, refined=refined)
                                                  
        img_diff,diff_points,diff_Flux = create_diff(img_diff,blobs_final,FWHM_MUSE=1.0)
        
        if (error == 1):
            map_now = np.sqrt(map_now)
            flux_HII = np.sqrt(flux_HII)
            diff_Flux = np.sqrt(diff_Flux)
            img_HII = np.sqrt(img_HII)
            img_diff = np.sqrt(img_diff)
        dict_HII[val_now] = flux_HII    
        dict_DIG[val_now] = diff_Flux
    
        model_HII_SSP[i,:,:] = img_HII
        model_diff_SSP[i,:,:] = img_diff
    
        if (plot == 1):
            for ax in axes:
                ax.cla()
            (nx_ssp,ny_ssp) = map_now.shape
            cmap='gist_stern_r'
            im_Ha=axes[0].imshow(map_now, interpolation='none',
                                         cmap=cmap, 
                                  norm=colors.PowerNorm(gamma=0.5)) 
            clim=im_Ha.properties()['clim']
            axes[1].imshow(img_HII, interpolation='none',
                           cmap=cmap, label=r'H$\alpha$',
                           norm=colors.PowerNorm(gamma=0.5), clim=clim) 
            axes[2].imshow(img_diff, interpolation='none',
                           cmap=cmap, label=r'H$\alpha$', 
                           norm=colors.PowerNorm(gamma=0.5), clim=clim) 

            axes[0].set_xlim(0,nx_ssp)
            axes[0].set_ylim(0,ny_ssp)
            axes[1].set_xlim(0,nx_ssp)
            axes[1].set_ylim(0,ny_ssp)    
            axes[2].set_xlim(0,nx_ssp)
            axes[2].set_ylim(0,ny_ssp)  
            axes[0].set_title(val_now)
            fig.canvas.draw()

    hdu_SSP_HII = fits.PrimaryHDU(data = model_HII_SSP , header = hdr_ssp)
    hdu_SSP_DIG = fits.PrimaryHDU(data = model_diff_SSP , header = hdr_ssp)

    table_SSP_HII = Table(dict_HII)
    for key in dict_HII.keys():
        table_SSP_HII[key].unit=dict_HII_units[key]


    table_SSP_DIG = Table(dict_DIG)
    for key in dict_DIG.keys():
        table_SSP_DIG[key].unit=dict_DIG_units[key]


    return hdu_SSP_HII, hdu_SSP_DIG, table_SSP_HII, table_SSP_DIG


def extracting_sfh(name, hdu_sfh, WCS, blobs_final, diff_points, plot=0):

	data_SFH=hdu_sfh[0].data
	hdr_sfh=hdu_sfh[0].header	
	
	keys = np.array(list(hdr_sfh.keys()))
	val = np.array(list(hdr_sfh.values()))
	
	nx_sfh = hdr_sfh['NAXIS1']
	ny_sfh = hdr_sfh['NAXIS2']
	nz_sfh = hdr_sfh['NAXIS3']

	model_HII_SFH = np.zeros((nz_sfh,ny_sfh,nx_sfh))
	model_diff_SFH = np.zeros((nz_sfh,ny_sfh,nx_sfh))

	dict_HII = {}
	list_HII = []

	for i in range(len(blobs_final)):
		list_HII.append(name+"-"+str(i+1))
	dict_HII['HIIREGID'] = list_HII
	dict_HII['X'] = blobs_final[:,1]
	dict_HII['Y'] = blobs_final[:,0]
	dict_HII['R'] = blobs_final[:,2]
	dict_HII['RA'] = (blobs_final[:,1]-WCS[0]+1)*WCS[4]+WCS[2]
	dict_HII['DEC'] = (blobs_final[:,0]-WCS[1]+1)*WCS[5]+WCS[3]
	dict_HII_units = {}
	dict_HII_units['HIIREGID'] = 'none'
	dict_HII_units['X'] = 'spaxels'
	dict_HII_units['Y'] = 'spaxels'
	dict_HII_units['R'] = 'spaxels'
	dict_HII_units['RA'] = 'deg'
	dict_HII_units['DEC'] = 'deg'

	dict_DIG = {}
	list_DIG = []

	for i in range(len(diff_points)):
		list_DIG.append(name+"-DIG-"+str(i+1))
	dict_DIG['DIGREGID'] = list_DIG
	dict_DIG['X'] = diff_points[:,0]
	dict_DIG['Y'] = diff_points[:,1]
	dict_DIG['RA'] = (diff_points[:,0]-WCS[0]+1)*WCS[4]+WCS[2]
	dict_DIG['DEC'] = (diff_points[:,1]-WCS[1]+1)*WCS[5]+WCS[3]
	dict_DIG_units = {}
	dict_DIG_units['DIGREGID'] = 'none'
	dict_DIG_units['X'] = 'spaxels'
	dict_DIG_units['Y'] = 'spaxels'
	dict_DIG_units['RA'] = 'deg'
	dict_DIG_units['DEC'] = 'deg'

	plot=0

	if (plot == 1):
		fig, axes = plt.subplots(1,3,figsize=(13,5))
		fig.canvas.set_window_title('HII explorer')
    
	for i in range(nz_sfh):
		key_now = 'DESC_'+str(i)
		val_now = hdr_sfh[key_now].replace('Luminosity Fraction for','LF_')   
               #val_now = hdr_sfh[i]
		dict_HII_units[val_now] = ''
		error = 0
		kind = 0
		refined = 0
		if ('LF_' in val_now):
			error=0
			refined=0
		dict_DIG_units[val_now] = dict_HII_units[val_now]
        #print(key_now,' = ',val_now,' , err= ',error,', kind=',kind)
        #print(key_now)
		map_now = data_SFH[i,:,:]
		if (error == 1):
			map_now = map_now**2
			
		flux_HII,img_HII,img_diff,diff_points,diff_val = HIIextraction(map_now,blobs_final,kind=kind, we=2, refined=refined)
		img_diff,diff_points,diff_Flux = create_diff(img_diff,blobs_final,FWHM_MUSE=1.0)
		
		if (error == 1):
			map_now = np.sqrt(map_now)
			flux_HII = np.sqrt(flux_HII)
			diff_Flux = np.sqrt(diff_Flux)
			img_HII = np.sqrt(img_HII)
			img_diff = np.sqrt(img_diff)
		dict_HII[val_now] = flux_HII    
		dict_DIG[val_now] = diff_Flux
    
		model_HII_SFH[i,:,:] = img_HII
		model_diff_SFH[i,:,:] = img_diff
    
		if (plot == 1):
			for ax in axes:
				ax.cla()
			(nx_sfh,ny_sfh) = map_now.shape
			cmap='gist_stern_r'
			im_Ha=axes[0].imshow(map_now, interpolation='none',\
                                         cmap=cmap, \
                                  norm=colors.PowerNorm(gamma=0.5)) 
			clim=im_Ha.properties()['clim']
			axes[1].imshow(img_HII, interpolation='none',\
                           cmap=cmap, label=r'H$\alpha$',\
                           norm=colors.PowerNorm(gamma=0.5), clim=clim) 
			axes[2].imshow(img_diff, interpolation='none',\
                           cmap=cmap, label=r'H$\alpha$', \
                           norm=colors.PowerNorm(gamma=0.5), clim=clim) 

			axes[0].set_xlim(0,nx_sfh)
			axes[0].set_ylim(0,ny_sfh)
			axes[1].set_xlim(0,nx_sfh)
			axes[1].set_ylim(0,ny_sfh)    
			axes[2].set_xlim(0,nx_sfh)
			axes[2].set_ylim(0,ny_sfh)  
			axes[0].set_title(val_now)
			fig.canvas.draw()

	hdu_SFH_HII = fits.PrimaryHDU(data = model_HII_SFH , header = hdr_sfh)
	hdu_SFH_DIG = fits.PrimaryHDU(data = model_diff_SFH , header = hdr_sfh)

	table_SFH_HII = Table(dict_HII)
	for key in dict_HII.keys():
		table_SFH_HII[key].unit=dict_HII_units[key]

	table_SFH_DIG = Table(dict_DIG)
	for key in dict_DIG.keys():
		table_SFH_DIG[key].unit=dict_DIG_units[key]
		

	return hdu_SFH_HII, hdu_SFH_DIG, table_SFH_HII, table_SFH_DIG
    

def extracting_index(name, hdu_index, WCS, blobs_final, diff_points, plot=0):

    desc_index = ["Hd", "Hb", "Mgb", "Fe5270", "Fe5335", "D4000", "Hdmod", "Hg", "SN", 
               "e_Hd", "e_Hb", "e_Mgb", "e_Fe5270", "e_Fe5335", "e_D4000", "e_Hdmod", 
               "e_Hg", "e_SN"]

    data_INDEX=hdu_index[0].data
    hdr_index=hdu_index[0].header
    hdr_index["INDEX5"] = "D_4000"
    
    keys = np.array(list(hdr_index.keys()))
    val = np.array(list(hdr_index.values()))

    nx_index = hdr_index['NAXIS1']
    ny_index = hdr_index['NAXIS2']
    nz_index = hdr_index['NAXIS3']

    model_HII_INDEX = np.zeros((nz_index,ny_index,nx_index))
    model_diff_INDEX = np.zeros((nz_index,ny_index,nx_index))

    dict_HII = {}
    list_HII = []

    for i in range(len(blobs_final)):
        list_HII.append(name+"-"+str(i+1))
    dict_HII['HIIREGID'] = list_HII
    dict_HII['X'] = blobs_final[:,1]
    dict_HII['Y'] = blobs_final[:,0]
    dict_HII['R'] = blobs_final[:,2]
    dict_HII['RA'] = (blobs_final[:,1]-WCS[0]+1)*WCS[4]+WCS[2]
    dict_HII['DEC'] = (blobs_final[:,0]-WCS[1]+1)*WCS[5]+WCS[3]
    dict_HII_units = {}
    dict_HII_units['HIIREGID'] = 'none'
    dict_HII_units['X'] = 'spaxels'
    dict_HII_units['Y'] = 'spaxels'
    dict_HII_units['R'] = 'spaxels'
    dict_HII_units['RA'] = 'deg'
    dict_HII_units['DEC'] = 'deg'
    
    dict_DIG = {}
    list_DIG = []
    
    for i in range(len(diff_points)):
        list_DIG.append(name+"-DIG-"+str(i+1))
    dict_DIG['DIGREGID'] = list_DIG
    dict_DIG['X'] = diff_points[:,0]
    dict_DIG['Y'] = diff_points[:,1]
    dict_DIG['RA'] = (diff_points[:,0]-WCS[0]+1)*WCS[4]+WCS[2]
    dict_DIG['DEC'] = (diff_points[:,1]-WCS[1]+1)*WCS[5]+WCS[3]
    dict_DIG_units = {}
    dict_DIG_units['DIGREGID'] = 'none'
    dict_DIG_units['X'] = 'spaxels'
    dict_DIG_units['Y'] = 'spaxels'
    dict_DIG_units['RA'] = 'deg'
    dict_DIG_units['DEC'] = 'deg'
    
    
    plot=0
    
    if (plot == 1):
    
        fig, axes = plt.subplots(1,3,figsize=(13,5))
        fig.canvas.set_window_title('HII explorer')
    
    #for i in range(5):
    for i in range(nz_index):
        key_now = 'INDEX'+str(i)
        val_now = hdr_index[key_now]
        dict_HII_units[val_now] = ''
        error = 0
        kind = 1
        refined = 0
        if ('e_' in val_now):
            error=1
            refined=0
        dict_DIG_units[val_now] = dict_HII_units[val_now]
        #print(key_now,' = ',val_now,' , err= ',error,', kind=',kind)
        map_now = data_INDEX[i,:,:]
        if (error == 1):
            map_now = map_now**2
        flux_HII,img_HII,img_diff,diff_points,diff_val = HIIextraction(map_now,blobs_final,kind=kind, we=2, refined=refined)
        img_diff,diff_points,diff_Flux = create_diff(img_diff,blobs_final,FWHM_MUSE=1.0)
        if (error == 1):
            map_now = np.sqrt(map_now)
            flux_HII = np.sqrt(flux_HII)
            diff_Flux = np.sqrt(diff_Flux)
            img_HII = np.sqrt(img_HII)
            img_diff = np.sqrt(img_diff)
        dict_HII[val_now] = flux_HII    
        dict_DIG[val_now] = diff_Flux
        
        model_HII_INDEX[i,:,:] = img_HII
        model_diff_INDEX[i,:,:] = img_diff
        
        if (plot == 1):
            for ax in axes:
                ax.cla()
            (nx_index,ny_index) = map_now.shape
            cmap='gist_stern_r'
            im_Ha=axes[0].imshow(map_now, interpolation='none',\
                                         cmap=cmap, \
                                  norm=colors.PowerNorm(gamma=0.5)) 
            clim=im_Ha.properties()['clim']
            axes[1].imshow(img_HII, interpolation='none',\
                           cmap=cmap, label=r'H$\alpha$',\
                           norm=colors.PowerNorm(gamma=0.5), clim=clim) 
            axes[2].imshow(img_diff, interpolation='none',\
                           cmap=cmap, label=r'H$\alpha$', \
                           norm=colors.PowerNorm(gamma=0.5), clim=clim) 

            axes[0].set_xlim(0,nx_index)
            axes[0].set_ylim(0,ny_index)
            axes[1].set_xlim(0,nx_index)
            axes[1].set_ylim(0,ny_index)    
            axes[2].set_xlim(0,nx_index)
            axes[2].set_ylim(0,ny_index)  
            axes[0].set_title(val_now)
            fig.canvas.draw()
    
    hdu_INDEX_HII = fits.PrimaryHDU(data = model_HII_INDEX , header = hdr_index)
    hdu_INDEX_DIG = fits.PrimaryHDU(data = model_diff_INDEX , header = hdr_index)

    table_INDEX_HII = Table(dict_HII)
    for key in dict_HII.keys():
        table_INDEX_HII[key].unit=dict_HII_units[key]
    
    table_INDEX_DIG = Table(dict_DIG)
    for key in dict_DIG.keys():
        table_INDEX_DIG[key].unit=dict_DIG_units[key]

    return hdu_INDEX_HII, hdu_INDEX_DIG, table_INDEX_HII, table_INDEX_DIG
    
def write_all(DIR, name, hdu_HII, hdu_DIG, table_HII, table_DIG, hdu_SSP_HII, hdu_SSP_DIG, table_SSP_HII, table_SSP_DIG,hdu_SFH_HII, hdu_SFH_DIG, table_SFH_HII, table_SFH_DIG, hdu_INDEX_HII, hdu_INDEX_DIG, table_INDEX_HII, table_INDEX_DIG):

    
    file_HII = DIR+"/HII."+name+".flux_elines.cube.fits.gz"
    file_DIG = DIR+"/DIG."+name+".flux_elines.cube.fits.gz"

    file_SSP_HII = DIR+"/HII."+name+'.SSP.cube.fits.gz'
    file_SSP_DIG = DIR+"/DIG."+name+'.SSP.cube.fits.gz'
    
    file_SFH_HII = DIR+"/HII."+name+'.SFH.cube.fits.gz'
    file_SFH_DIG = DIR+"/DIG."+name+'.SFH.cube.fits.gz'

    file_INDEX_HII = DIR+"/HII."+name+'.INDEX.cube.fits.gz'
    file_INDEX_DIG = DIR+"/DIG."+name+'.INDEX.cube.fits.gz'

    hdu_HII.writeto(file_HII,  overwrite=True)
    hdu_DIG.writeto(file_DIG, overwrite=True)
    
    hdu_SSP_HII.writeto(file_SSP_HII, overwrite=True)
    hdu_SSP_DIG.writeto(file_SSP_DIG, overwrite=True)
    
    hdu_SFH_HII.writeto(file_SFH_HII, overwrite=True)
    hdu_SFH_DIG.writeto(file_SFH_DIG, overwrite=True)
    
    #hdr_index["INDEX5"] = "D_4000"
    
    hdu_INDEX_HII.writeto(file_INDEX_HII, output_verify="ignore", overwrite=True)
    hdu_INDEX_DIG.writeto(file_INDEX_DIG, output_verify="ignore", overwrite=True)
    

    file_table_HII = DIR+"/HII."+name+".flux_elines.table.fits.gz"
    file_table_DIG = DIR+"/DIG."+name+".flux_elines.table.fits.gz"
    
    file_table_SSP_HII = DIR+"/HII."+name+'.SSP.table.fits.gz'
    file_table_SSP_DIG = DIR+"/DIG."+name+'.SSP.table.fits.gz'
    
    file_table_SFH_HII = DIR+"/HII."+name+'.SFH.table.fits.gz'
    file_table_SFH_DIG = DIR+"/DIG."+name+'.SFH.table.fits.gz'
    
    file_table_INDEX_HII = DIR+"/HII."+name+'.INDEX.table.fits.gz'
    file_table_INDEX_DIG = DIR+"/DIG."+name+'.INDEX.table.fits.gz'


    table_HII.write(file_table_HII, overwrite=True)
    table_DIG.write(file_table_DIG, overwrite=True)
    
    table_SSP_HII.write(file_table_SSP_HII, overwrite=True)
    table_SSP_DIG.write(file_table_SSP_DIG, overwrite=True)
    
    table_SFH_HII.write(file_table_SFH_HII, overwrite=True)
    table_SFH_DIG.write(file_table_SFH_DIG, overwrite=True)
    
    table_INDEX_HII.write(file_table_INDEX_HII, overwrite=True)
    table_INDEX_DIG.write(file_table_INDEX_DIG, overwrite=True)
   
    print("Finish writing"+ " in "+DIR)  

################################################################################################
# SFS & JKBB
# * END
################################################################################################


################################################################################################
# AZLA
# Date: 
# * STARTS
################################################################################################


################################################################################################
# AZLA
# * END
################################################################################################ 
