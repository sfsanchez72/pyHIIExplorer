################################################################################################
#
# HIIblob
#
# Description:
#
# Date: 04.11.2020
#
# Authors: A.Z. Lugo-Aranda , S.F. Sanchez & J.K. Barrera-Ballesteros
#
################################################################################################


################################################################################################
# SFS & JKBB
# * STARS
# DATE: Sep-Nov
################################################################################################

import numpy as np
import scipy as sp
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
#matplotlib.interactive(False)
import matplotlib.pyplot as plt
#matplotlib.interactive(True)

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
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import warnings
#
# Core functions to run HIIblob
#

def forward(x):
    return x**(1/2)


def inverse(x):
    return x**2

def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


#
# Extraction with an interative mapping
#
def extract_flux_refined(blobs,flux_input,image,kind=0,we=0,dr=3,plot=0):
# kind = 0 SUM
# kind = 1 MEAN
# kind = 2 VAR

# we = 0 No Weight
# we = 1 Gaussian Weight
# we = 2 Gaussian quadration Weight
    n=len(blobs)
    (nx,ny) = image.shape
    image = np.ma.masked_invalid(image)
    flux = np.zeros(n)
    i=0
    
    if (plot==1):
        fig = plt.figure(figsize=(8,8))
        fig.canvas.set_window_title('Canvas active title')
        ax2 = fig.add_subplot(111)
        cmap='gist_stern_r'
        im_Ha_MUSE=ax2.imshow(image, interpolation='none',\
                              cmap=cmap, label=r'H$\alpha$',\
                              norm=colors.PowerNorm(gamma=0.15)) 
        clim=im_Ha_MUSE.properties()['clim']
        ax2.set_xlim(0,nx)
        ax2.set_ylim(0,ny)
        ax1 = fig.add_axes([0.15, 0.9, 0.17, 0.08]) 
        ax3 = fig.add_axes([0.30, 0.9, 0.17, 0.08]) 
        ax4 = fig.add_axes([0.45, 0.9, 0.17, 0.08]) 
        
        
#        ax1 = fig.add_subplot(111)
#        ax2 = fig.add_subplot(121)
#    print('blobs[0]=',blobs[0], blobs[0][0], blobs[0][1], blobs[0][2])
#    print('blobs[:,0]=',blobs[:,0])
#    print('blobs[:][1]=',blobs[:,1])
    for blob in blobs:
#    n=len(blobs)
        y, x, r = blob
        flux_now = 0
        i0=int(x-dr*r)
        i1=int(x+dr*r)
        j0=int(y-dr*r)
        j1=int(y+dr*r)
        if (i0<0):
            i0=0
        if (j0<0):
            j0=0
        if (i1>(nx-1)):
            i1=nx-1
        if (j1>(ny-1)):
            j1=ny-1
         

        #
        # Blobs around!
        #
        mask_blobs_sec = (blobs[:,1]>=i0) & (blobs[:,1]<=i1) & \
        (blobs[:,0]>=j0) & (blobs[:,0]<=j1) & \
        (blobs[:,1]!=x) & (blobs[:,0]!=y) 
#        print (mask_blobs_sec)        
#        print('X,Y blob = ',x,y,i0,i1,j0,j1)
        ind_blobs = np.where(mask_blobs_sec)
        blobs_near=blobs[ind_blobs]
        flux_input_near=flux_input[ind_blobs]
        blobs_near[:,1]=blobs_near[:,1]-i0
        blobs_near[:,0]=blobs_near[:,0]-j0
#        print('Blobs near =',blobs[ind_blobs])
        #blobs_sec = 
            
        image_sec=image[j0:j1,i0:i1]        
        (ny_sec,nx_sec)=image_sec.shape

        image_sec_model=create_HII_image(blobs_near,flux_input_near,nx_sec,ny_sec,dr=dr)
        image_sec_clean=image_sec-image_sec_model
        x_g = np.arange(0,nx_sec,1)
        y_g = np.arange(0,ny_sec,1)
        x_g, y_g = np.meshgrid(x_g, y_g) # get 2D variables instead of 1D
        xp_g = x-i0
        yp_g = y-j0
#
# No weight!
#
        w_l = np.ones((ny_sec,nx_sec))
        dist = np.sqrt((x_g-xp_g)**2+(y_g-yp_g)**2)
        w_l[dist>r]=0
        
#
# r<r_HII -> w_l = 0
#

        w_g = gaus2d(x_g, y_g, mx=xp_g, my=yp_g,sx=r, sy=r)
        w_g[dist>2*r]=0
        
# WE = 2
        w_q = w_g**4
        w_q[dist>2*r]=0
        
        if (we==0):
            w=w_l
        if (we==1):
            w=w_g
        if (we==2):
            w=w_q
            
        #        print(w_g)
        #print(w.shape,w_g.shape)
        #
        # SUM
        #
        if (kind==0):
            flux_now=np.ma.sum(image_sec_clean*w)
            w_now=np.sum(w)
            flux_now /= w_now
            flux_now *= np.sum(w_l)#(nx_sec*ny_sec)
        #
        # MEAN
        #
        if (kind==1):
            flux_now=np.ma.sum(image_sec_clean*w)
            w_now=np.sum(w)
            flux_now /= w_now
        #
        # STDDED
        #
        flux[i]=flux_now
        i=i+1


        if (plot==1):            
            ax1.cla()
            im_sec=ax1.imshow(image_sec, interpolation='none',\
                                     cmap=cmap, label=r'H$\alpha$',\
                                     norm=colors.PowerNorm(gamma=0.15),\
                                     clim=clim) 
            mark_HII = ax1.plot(x-i0,y-j0,'+', color='white', alpha=0.25)
#            mark_near = ax1.plot(blobs_near[:,1],blobs_near[:,0],'ro', color='white', alpha=0.25)            
            nx_sec = j1-j0
            ny_sec = i1-i0
            ax1.set_xlim(0,nx_sec)
            ax1.set_ylim(0,ny_sec)
            
            im_sec_model=ax3.imshow(image_sec_model, interpolation='none',\
                                    cmap=cmap, label=r'H$\alpha$',\
                                    norm=colors.PowerNorm(gamma=0.15),\
                                    clim=clim) 
            ax3.set_xlim(0,nx_sec)
            ax3.set_ylim(0,ny_sec)
            
            im_sec_model=ax4.imshow(image_sec_clean, interpolation='none',\
                                    cmap=cmap, label=r'H$\alpha$',\
                                    norm=colors.PowerNorm(gamma=0.15),\
                                    clim=clim) 
            ax4.set_xlim(0,nx_sec)
            ax4.set_ylim(0,ny_sec)
            mark = ax2.plot(x,y,'ro', color='white', alpha=0.25) #scatter(x,y,alpha=0.2)
            
            fig.canvas.draw()
#            time.sleep(3)

#            for blob in blobs_final:
#                y, x, r = blob
#                c = plt.Circle((x, y), r, color='darkred', linewidth=2, fill=False, alpha=0.25)
#                axes[0][0].add_patch(c)
        
        
    return flux

def extract_flux(blobs,image,kind=0,we=0,dr=3):
# kind = 0 SUM
# kind = 1 MEAN
# kind = 2 VAR

# we = 0 No Weight
# we = 1 Gaussian Weight
# we = 2 Gaussian quadration Weight
    n=len(blobs)
    (nx,ny) = image.shape
    image = np.ma.masked_invalid(image)
    flux = np.zeros(n)
    i=0
    for blob in blobs:
        y, x, r = blob
        flux_now = 0
        i0=int(x-dr*r)
        i1=int(x+dr*r)
        j0=int(y-dr*r)
        j1=int(y+dr*r)
        if (i0<0):
            i0=0
        if (j0<0):
            j0=0
        if (i1>(nx-1)):
            i1=nx-1
        if (j1>(ny-1)):
            j1=ny-1
        image_sec=image[j0:j1,i0:i1] 
        (ny_sec,nx_sec)=image_sec.shape
        x_g = np.arange(0,nx_sec,1)
        y_g = np.arange(0,ny_sec,1)
        x_g, y_g = np.meshgrid(x_g, y_g) # get 2D variables instead of 1D
        xp_g = x-i0
        yp_g = y-j0
#
# No weight!
#
        w_l = np.ones((ny_sec,nx_sec))
        dist = np.sqrt((x_g-xp_g)**2+(y_g-yp_g)**2)
        w_l[dist>r]=0
        
#
# r<r_HII -> w_l = 0
#

        w_g = gaus2d(x_g, y_g, mx=xp_g, my=yp_g,sx=r, sy=r)
        w_g[dist>2*r]=0
        
# WE = 2
        w_q = w_g**4
        w_q[dist>2*r]=0
        
        if (we==0):
            w=w_l
        if (we==1):
            w=w_g
        if (we==2):
            w=w_q
            
        #        print(w_g)
        #print(w.shape,w_g.shape)
        #
        # SUM
        #
        if (kind==0):
            flux_now=np.ma.sum(image_sec*w)
            w_now=np.sum(w)
            flux_now/= w_now
            flux_now *= np.sum(w_l)#(nx_sec*ny_sec)
#            flux_now = flux_now*np.sqrt(2)
        #
        # MEAN
        #
        if (kind==1):
            flux_now=np.ma.sum(image_sec*w)
            w_now=np.sum(w)
            flux_now /= w_now
        #
        # STDDED
        #
        flux[i]=flux_now
        i=i+1
    return flux

#fluxMUSE_1sig_V=np.min(np.array([MUSE_1sig_V,median_V_MUSE_pos,std_V_MUSE_pos,MUSE_1sig]))
#MUSE_1sig_V=np.min(np.array([MUSE_1sig_V,std_V_MUSE_pos]))
#print('1sig MUSE-V ',MUSE_1sig_V)

def extract_flux_points(points,r_points,image,kind=0,we=0,dr=3):
    n=len(r_points)
    blobs=np.zeros((n,3))
    blobs[:,0]=points[:,0]
    blobs[:,1]=points[:,1]
    blobs[:,2]=r_points
    flux=extract_flux(blobs,image,kind=kind,we=we,dr=dr)
    #
    # We mask infinites and nan
    #
    return flux
    
def extract_flux_points_OLD(points,r_points,image,kind=0,we=0,dr=3):
# kind = 0 SUM
# kind = 1 MEAN
# kind = 2 VAR

# we = 0 No Weight
# we = 1 Gaussian Weight
# we = 2 Gaussian to the power of 2 Weight
    n=len(points)
    (nx,ny) = image.shape
    image = np.ma.masked_invalid(image)
    flux = np.zeros(n)
    i=0
    for blob,r in zip(points,r_points):
        y, x = blob
        flux_now = 0
        i0=int(x-dr*r)
        i1=int(x+dr*r)
        j0=int(y-dr*r)
        j1=int(y+dr*r)
        if (i0<0):
            i0=0
        if (j0<0):
            j0=0
        if (i1>(nx-1)):
            i1=nx-1
        if (j1>(ny-1)):
            j1=ny-1
        image_sec=image[j0:j1,i0:i1] 
        (ny_sec,nx_sec)=image_sec.shape
        x_g = np.arange(0,nx_sec,1)
        y_g = np.arange(0,ny_sec,1)
        x_g, y_g = np.meshgrid(x_g, y_g) # get 2D variables instead of 1D
        xp_g = x-i0
        yp_g = y-j0
        w_l = np.ones((ny_sec,nx_sec))
        dist = np.sqrt((x_g-xp_g)**2+(y_g-yp_g)**2)
        w_l[dist>r]=0
        
#
# r<r_HII -> w_l = 0
#

        w_g = gaus2d(x_g, y_g, mx=xp_g, my=yp_g,sx=r, sy=r)
        w_g[dist>2*r]=0
        
# WE = 2
        w_q = w_g**4
        w_q[dist>2*r]=0
        
        if (we==0):
            w=w_l
        if (we==1):
            w=w_g
        if (we==2):
            w=w_q
            
        #        print(w_g)
        #print(w.shape,w_g.shape)
        if (kind==0):
            flux_now=np.ma.sum(image_sec*w)
            w_now=np.sum(w)
            flux_now /= w_now
            flux_now *= (nx_sec*ny_sec)
        if (kind==1):
            flux_now=np.ma.sum(image_sec*w)
            w_now=np.sum(w)
            flux_now /= w_now
        flux[i]=flux_now
        i=i+1
    return flux

#def create_HII_seg(blobs,nx,ny):
#   image = np.zeros((ny,nx))
#   for blob in blobs:
#        y, x, r = blob
#        dist = 
#       Matrix = {(x,y):0 for x in range(nx) for y in range(ny)}

def create_HII_image(blobs,flux,nx,ny,dr=5):
    image = np.zeros((ny,nx))
    for blob,bflux in zip(blobs,flux):
        y, x, r = blob
        i0=int(x-dr*r)
        i1=int(x+dr*r)
        j0=int(y-dr*r)
        j1=int(y+dr*r)
        if (i0<0):
            i0=0
        if (j0<0):
            j0=0
        if (i1>(nx-1)):
            i1=nx-1
        if (j1>(ny-1)):
            j1=ny-1
#        print(x,y,' sec y =',j0,j1,' sec x =',i0,i1,' flux=',bflux)
        image_sec=image[j0:j1,i0:i1] 
        (ny_sec,nx_sec)=image_sec.shape
        x_g = np.arange(0,nx_sec,1)
        y_g = np.arange(0,ny_sec,1)        
        x_g, y_g = np.meshgrid(x_g, y_g) # get 2D variables instead of 1D
        xp_g = x-i0
        yp_g = y-j0
#        w_g = gaus2d(x_g, y_g, mx=xp_g, my=yp_g,sx=r/sqrt(1.44), sy=r/sqrt(1.44))
        w_g = sqrt(2)*gaus2d(x_g, y_g, mx=xp_g, my=yp_g,sx=r, sy=r)
#        w_g = gaus2d(x_g, y_g, sx=r, sy=r)
#        w_g = gaus2d(x_g, y_g, mx=xp_g, my=yp_g,sx=r/sqrt(2), sy=r/sqrt(2))
#        w_g = gaus2d(x_g, y_g, mx=xp_g, my=yp_g,sx=r*sqrt(2), sy=r*sqrt(2))

        image_HII = bflux*w_g * sqrt(2)#/ (2. * np.pi )
#        (2*np.pi)
        if ((~np.isnan(bflux)) and (bflux>0)):
            image[j0:j1,i0:i1]=image[j0:j1,i0:i1]+image_HII
    return image


def delta_HII_image_to_fit(org_img,blobs,nx,ny,dr=5): 
    def delta_img(x, *flux): 
        model_img=create_HII_image(blobs,flux,nx,ny,dr=dr)
        delta = np.nansum((org_img-model_img)**2/org_img)
        return delta
    return delta_img

def HII_image_to_fit(blobs,nx,ny,dr=5): 
    def delta_img(x, *flux): 
        model_img=create_HII_image(blobs,flux,nx,ny,dr=dr)
        return model_img
    return delta_img

def flatten_HII_image_to_fit(blobs,nx,ny,dr=5): 
    def delta_img(x, *flux): 
        model_img=create_HII_image(blobs,flux,nx,ny,dr=dr)
        return model_img.flatten()
    return delta_img

def ravel_HII_image_to_fit(blobs,nx,ny,dr=5): 
    def delta_img(x, *flux): 
        model_img=create_HII_image(blobs,flux,nx,ny,dr=dr)
        return model_img.ravel()
    return delta_img

def ravel_Gauss_HII_img(xdata, *flux):
    global BLOBS
    global NX
    global NY
    global DR
    model_img=create_HII_image(BLOBS,flux,NX,NY,dr=DR)
    return model_img.ravel()


#
# We need an Halpha and a continuum image to detect the Hii regions
#
#spax_sca_MUSE = 0.2
#FWHM_MUSE = 1.0
#FWHM_MUSE = FWHM_MUSE/spax_sca_MUSE


def create_diff(Ha_image,blobs_log_MUSE,FWHM_MUSE):
    #
    # We select those regions far away from the Hii regions
    #
    (nx,ny)=Ha_image.shape
    points_MUSE=blobs_log_MUSE[:,0:2]
#    print("LEN_DIFF = ",len(points_MUSE)," DIFF=",points_MUSE)
    if(len(points_MUSE)>3):
        tri_MUSE = Delaunay(points_MUSE)
        diff_p_MUSE = np.zeros((len(tri_MUSE.simplices),2))
        r_p_MUSE = np.zeros(len(tri_MUSE.simplices))
        i=0
        for j, s in enumerate(tri_MUSE.simplices):
            try:
                p = points_MUSE[s].mean(axis=0)
                diff_p_MUSE[s,0]=p[0]
                diff_p_MUSE[s,1]=p[1]
            except:
                print("no diffuse points");
        #
        # We remove repeated points in the diffuse 
        #
        diff_p_MUSE = unique2d(diff_p_MUSE)
        #
        # We remove the diffuse too near to an Hii region
        #
        dist_HII_diff,index_HII_diff = do_kdtree(blobs_log_MUSE[:,0:2],diff_p_MUSE)
        #    mask_HII_diff = dist_HII_diff>2*blobs_log_MUSE[index_HII_diff,2]
        #mask_HII_diff = dist_HII_diff>2*blobs_log_MUSE[index_HII_diff,2]
        mask_HII_diff = dist_HII_diff>1.5*blobs_log_MUSE[index_HII_diff,2]
        diff_p_MUSE = diff_p_MUSE[mask_HII_diff]
        #
        # We extract aperture of FWHMN (mean), around the diffse points
        #
        r_p_MUSE=FWHM_MUSE*np.ones(len(diff_p_MUSE))
        F_Ha_diff_MUSE=extract_flux_points(diff_p_MUSE,r_p_MUSE,Ha_image,kind=1,we=2)    
        xgrid=np.arange(0, ny, 1)
        ygrid=np.arange(0, nx, 1)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        #print("DIFF=",len(diff_p_MUSE[:,1]),len(diff_p_MUSE[:,0]))
        if (len(diff_p_MUSE[:,1])>3):
            new_array=np.array((diff_p_MUSE[:,1],diff_p_MUSE[:,0])).T
            diff_interp_map = griddata(new_array, F_Ha_diff_MUSE, (Xgrid, Ygrid), method='nearest')
            diff_interp_map_g=gaussian_filter(diff_interp_map, sigma=FWHM_MUSE)
        else:
            diff_interp_map_g=gaussian_filter(Ha_image, sigma=3*FWHM_MUSE)
            diff_p_MUSE = np.zeros((len(points_MUSE),2))
            F_Ha_diff_MUSE = np.zeros(len(points_MUSE))
    else:
        diff_interp_map_g=gaussian_filter(Ha_image, sigma=3*FWHM_MUSE)
        diff_p_MUSE = np.zeros((len(points_MUSE),2))
        F_Ha_diff_MUSE = np.zeros(len(points_MUSE))
    return diff_interp_map_g,diff_p_MUSE,F_Ha_diff_MUSE

def HIIextraction(Ha_image,blobs_log_MUSE,kind=0,we=2,\
                  FWHM_MUSE = 1.0, refined = 0):
    """
    
    """
    (nx,ny)=Ha_image.shape
    #
    # Extract flux of Hii regions!
    #
    blobs_F_Ha=extract_flux(blobs_log_MUSE,Ha_image,kind=kind,we=we)    
    #
    # Only finite values
    #
    #mask_fin = np.isfinite(blobs_F_Ha)
    #blobs_log_MUSE = blobs_log_MUSE[mask_fin]
    #blobs_F_Ha = blobs_F_Ha[mask_fin]
    #
    # Create image of HII_regions
    #
    image_HII=create_HII_image(blobs_log_MUSE,blobs_F_Ha,ny,nx)
    #
    # Image clean from HII model
    #
    image_clean_HII = Ha_image-image_HII
    image_clean_HII_g=gaussian_filter(image_clean_HII, sigma=FWHM_MUSE)    
#    diff_interp_map_g=create_diff(Ha_image,blobs_log_MUSE,FWHM_MUSE)
    diff_interp_map_g,diff_points,diff_Flux  = create_diff(image_clean_HII_g,blobs_log_MUSE,FWHM_MUSE)
    i_ref=0
    while (i_ref<refined):
        F_Ha_MUSE_clean = Ha_image - diff_interp_map_g
        blobs_F_Ha=extract_flux_refined(blobs_log_MUSE,blobs_F_Ha,F_Ha_MUSE_clean,kind=0,we=we, plot=0) 
        #
        # Create image of HII_regions
        #
        image_HII=create_HII_image(blobs_log_MUSE,blobs_F_Ha,ny,nx)
        #
        # Image clean from HII model
        #
        image_clean_HII = Ha_image-image_HII
        image_clean_HII_g=gaussian_filter(image_clean_HII, sigma=FWHM_MUSE)    
        diff_interp_map_g,diff_points,diff_Flux=create_diff(image_clean_HII_g,blobs_log_MUSE,FWHM_MUSE)
        i_ref = i_ref+1
        
    return blobs_F_Ha,image_HII,diff_interp_map_g,diff_points,diff_Flux

def HIIdetection(Ha_image,min_sigma=0.8, max_sigma=2.0, num_sigma=30, threshold=20, FWHM_MUSE=1.0):
    """
    
    """
    (nx,ny)=Ha_image.shape
    #
    # Detect Hii regions!
    #
    blobs_log_MUSE = blob_log(Ha_image, min_sigma=min_sigma,\
                          max_sigma=max_sigma,\
                          num_sigma=num_sigma, threshold=threshold)
    blobs_log_MUSE[:, 2] = blobs_log_MUSE[:, 2] * sqrt(2) #1.75 #sqrt(2) #1.75 #sqrt(2)
    #
    # Extract flux of Hii regions!
    #
    blobs_F_Ha=extract_flux(blobs_log_MUSE,Ha_image,kind=0,we=2)    

    #
    # We mask infinites and nan
    #
    mask_fin = (np.isfinite(blobs_F_Ha)) & (blobs_F_Ha>0) 
    blobs_log_MUSE = blobs_log_MUSE[mask_fin]
    blobs_F_Ha = blobs_F_Ha[mask_fin]
    
    #
    # Create image of HII_regions
    #
    image_HII=create_HII_image(blobs_log_MUSE,blobs_F_Ha,ny,nx)
    #
    # Image clean from HII model
    #
    image_clean_HII = Ha_image-image_HII
    image_clean_HII_g=gaussian_filter(image_clean_HII, sigma=2*FWHM_MUSE)    
    
    diff_interp_map_g,diff_points,diff_Flux=create_diff(image_clean_HII_g,blobs_log_MUSE,FWHM_MUSE)

    #print('# Diff = ',len(diff_p_MUSE))
    return blobs_log_MUSE,blobs_F_Ha,image_HII,diff_interp_map_g

def HIIblob(F_Ha_MUSE,V_MUSE,FWHM_MUSE, MUSE_1sig=0, MUSE_1sig_V=0, plot=0, refined=0, name="NONE", DIG_lim=1, num_sigma=30):
    #min_sigma=0.8, max_sigma=2.0,\
    #                            num_sigma=30,):
    """
        Parameters
        ----------
        F_Ha_MUSE : 2D array image 
            The Ha intensity map
        V_MUSE : 2D array image
            Continuum map image
        FWHM : float
            FWHM of the image in arcsec

            
    """
    (nx,ny)=F_Ha_MUSE.shape
#
# Ha map statistics
#
    if (MUSE_1sig == 0):
        mask_F_Ha_neg = F_Ha_MUSE<0
        F_Ha_MUSE_neg = F_Ha_MUSE [mask_F_Ha_neg]
        F_Ha_MUSE_noise = np.concatenate((F_Ha_MUSE_neg,(-1)*F_Ha_MUSE_neg))
        mean_F_Ha_MUSE_noise = np.nanmean(F_Ha_MUSE_noise)
        std_F_Ha_MUSE_noise = np.nanstd(F_Ha_MUSE_noise)
        MUSE_1sig = std_F_Ha_MUSE_noise
    MUSE_3sig = 3*MUSE_1sig

# MUSE
# g-band
    if (MUSE_1sig_V == 0):
        mask_V_neg = V_MUSE<0
        V_MUSE_neg = V_MUSE [mask_V_neg]
        V_MUSE_pos = V_MUSE [~mask_V_neg]
        V_MUSE_noise = np.concatenate((V_MUSE_neg,(-1)*V_MUSE_neg))
        mean_V_MUSE_noise = np.nanmean(V_MUSE_noise)
        std_V_MUSE_noise = np.nanstd(V_MUSE_noise)
        MUSE_1sig_V = std_V_MUSE_noise
        median_V_MUSE_pos = np.median(V_MUSE_pos)
        std_V_MUSE_pos = np.nanstd(V_MUSE[V_MUSE<median_V_MUSE_pos])
        median_V_MUSE = np.median(V_MUSE[V_MUSE<median_V_MUSE_pos])
        MUSE_1sig_V=np.min(np.array([MUSE_1sig_V,median_V_MUSE_pos,std_V_MUSE_pos,MUSE_1sig]))

    MUSE_3sig_V = 3*MUSE_1sig

        
    if (np.isnan(MUSE_1sig_V)):
        MUSE_1sig_V = MUSE_1sig
        
    print('# 1sig Ha-map = ',MUSE_1sig,'; 1sig MUSE-V ',MUSE_1sig_V)


        
    #
    # We clean the invalid points
    #
    F_Ha_MUSE = np.ma.masked_invalid(F_Ha_MUSE)
    V_MUSE = np.ma.masked_invalid(V_MUSE)
    F_Ha_MUSE_g=gaussian_filter(F_Ha_MUSE, sigma=2*FWHM_MUSE)
    V_MUSE_g=gaussian_filter(V_MUSE, sigma=2*FWHM_MUSE)
    mask_MUSE = (F_Ha_MUSE_g<2*MUSE_1sig) | (V_MUSE_g<0.5*MUSE_1sig_V) 
    F_Ha_MUSE_masked = np.ma.array(F_Ha_MUSE, mask = mask_MUSE, fill_value=0.0)
    F_Ha_MUSE_fill = F_Ha_MUSE_masked.filled()

    #
    # Initial detection
    #
    blobs_log_MUSE,blobs_F_Ha,image_HII,diff_map=HIIdetection(F_Ha_MUSE_fill, min_sigma=0.8,\
                                                              max_sigma=FWHM_MUSE,\
                                                              num_sigma=num_sigma, threshold=1.5*MUSE_1sig,\
                                                              FWHM_MUSE = 1.0)

    #                                                               max_sigma=2.0*FWHM_MUSE,\

    print('# HII reg. Initial = ',len(blobs_log_MUSE))
    #
    # We now subtract the diffuse and we detect HII regions again!
    #
    F_Ha_MUSE_clean = F_Ha_MUSE_fill-diff_map
    F_Ha_MUSE_masked = np.ma.array(F_Ha_MUSE_clean, mask = mask_MUSE, fill_value=0.0)
    F_Ha_MUSE_fill = F_Ha_MUSE_masked.filled()    
    blobs_log_MUSE,blobs_F_Ha,image_HII,diff_map_2=HIIdetection(F_Ha_MUSE_fill, min_sigma=0.8,\
                                                                max_sigma=FWHM_MUSE,\
                                                                num_sigma=num_sigma, threshold=2.0*MUSE_1sig)
    print('# HII reg. 2nd = ',len(blobs_log_MUSE))
    diff_map,diff_points,diff_Flux = create_diff(F_Ha_MUSE_fill,blobs_log_MUSE,FWHM_MUSE)    
    res_map = F_Ha_MUSE-(image_HII+diff_map)
    
    #
    # We find extra regions?
    #
    blobs_log_MUSE_add,blobs_F_Ha_add,\
    image_HII_add,diff_map_add=HIIdetection(res_map, min_sigma=0.8, max_sigma=FWHM_MUSE,\
                                            num_sigma=num_sigma, threshold=5.0*MUSE_1sig)
    print('# HII reg. additional = ',len(blobs_log_MUSE_add))
    #  
    # Look for the 2nd more near point
    #
    dist_HII_self,index_HII_self = do_kdtree(blobs_log_MUSE[:,0:2],blobs_log_MUSE_add[:,0:2],k=1)
    mask_HII_self = (dist_HII_self>blobs_log_MUSE_add[:,2]) 
    blobs_log_MUSE_add = blobs_log_MUSE_add[mask_HII_self]        
    blobs_final=np.concatenate((blobs_log_MUSE,blobs_log_MUSE_add), axis=0)
    

    
    #
    # Final Extraction
    #
    blobs_F_Ha,image_HII,diff_map_final,diff_points,diff_Flux=HIIextraction(F_Ha_MUSE_fill,blobs_final,\
                                                                            kind=0,we=2,\
                                                              FWHM_MUSE = FWHM_MUSE, refined = 0)

    print("# Last iteration Num. = ",len(blobs_F_Ha))
    #
    # We clean the sample
    #

    #
    # Value of the diffuse at the location of Hii regions!
    #
    DIG_blobs_F_Ha,DIG_image_HII,DIG_diff_map_final,DIG_diff_points,DIG_diff_Flux=HIIextraction(diff_map_final,blobs_final,\
                                                                                                kind=0,we=2,\
                                                                FWHM_MUSE = FWHM_MUSE, refined = 0)

    

    mask_F_Ha = (blobs_F_Ha>MUSE_3sig) & (blobs_F_Ha>DIG_lim*DIG_blobs_F_Ha)
    blobs_F_Ha = blobs_F_Ha[mask_F_Ha]
    blobs_final = blobs_final[mask_F_Ha]
    print("# Clean above 3sigma Num = ",len(blobs_F_Ha))

    order=np.argsort(-1*blobs_F_Ha)
    blobs_F_Ha = blobs_F_Ha[order]
    blobs_final = blobs_final[order]


    
    blobs_F_Ha,image_HII,diff_map_final,diff_points,diff_Flux=HIIextraction(F_Ha_MUSE_fill,blobs_final,\
                                                                            kind=0,we=2,\
                                                              FWHM_MUSE = FWHM_MUSE, refined = refined)



    

    
    
    res_map = F_Ha_MUSE-(image_HII+diff_map)
    X_sqr = (res_map**2)/MUSE_3sig**2 #np.abs(F_Ha_MUSE)
    X_sqr_masked = np.ma.array(X_sqr, mask = mask_MUSE, fill_value=0.0)
 #   (nx_m,ny_m)=X_sqr.shape
    X_sqr_sum = np.ma.sum(X_sqr_masked)/X_sqr_masked.count()#len(X_sqr#(nx*ny)#len(blobs_F_Ha))
    print('# X_sqr = ',X_sqr_sum)
    print('# HII reg clean=',len(blobs_final))
    
    #
    # Lets make some plots!
    #
    if (plot==1):
#        matplotlib.rcParams['interactive'] == True
#        plt.ion()
        fig, axes = plt.subplots(2,2, figsize=(15,15))
        cmap='gist_stern_r'
        im_Ha_MUSE=axes[0][0].imshow(F_Ha_MUSE, interpolation='none',\
                                     cmap=cmap, label=r'H$\alpha$',\
                                     norm=colors.PowerNorm(gamma=0.5)) 
        for blob in blobs_final:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='darkred', linewidth=2, fill=False, alpha=0.25)
            axes[0][0].add_patch(c)
#        for blob in blobs_log_MUSE:
#            y, x, r = blob
#            c = plt.Circle((x, y), r, color='black', linewidth=2, fill=False, alpha=0.15)
#           axes[0][0].add_patch(c)
#        axes[0][0].scatter(blobs_log_MUSE[:,1],blobs_log_MUSE[:,0],s=blobs_log_MUSE[:,2]*20,\
#                           marker='+',edgecolor='blue',color='black',alpha=0.3)
        
        clim=im_Ha_MUSE.properties()['clim']
        im_HII_mod=axes[0][1].imshow(image_HII, interpolation='none',\
                                     cmap=cmap, label=r'Hii regions',\
                                     norm=colors.PowerNorm(gamma=0.5),clim=clim) 
        im_diff=axes[1][0].imshow(diff_map, interpolation='none',\
                                     cmap=cmap, label=r'Diffuse',\
                                  norm=colors.PowerNorm(gamma=0.5),clim=clim) 
        im_res=axes[1][1].imshow(res_map, interpolation='none',\
                                     cmap=cmap, label=r'Diffuse',\
                                 norm=colors.PowerNorm(gamma=0.5),clim=clim) 
        axes[0][0].set_xlim(0,nx)
        axes[0][0].set_ylim(0,ny)
        axes[0][1].set_xlim(0,nx)
        axes[0][1].set_ylim(0,ny)
        axes[1][0].set_xlim(0,nx)
        axes[1][0].set_ylim(0,ny)
        axes[1][1].set_xlim(0,nx)
        axes[1][1].set_ylim(0,ny)
        fileout_png = "HIIblob."+name+".png"
        fig.savefig(fileout_png, transparent=False, facecolor='white', edgecolor='white')

    return blobs_final,blobs_F_Ha,image_HII,diff_map_final,diff_points,diff_Flux


def unique2d(a):
    x, y = a.T
    b = x + y*1.0j 
    idx = np.unique(b,return_index=True)[1]
    return a[idx] 

def do_kdtree(combined_x_y_arrays,points,k=1):
    mytree = cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points,k=k)
    return np.array(dist),np.array(indexes)


def create_diff_new(Ha_image_clean,blobs,FWHM_MUSE,diff_points, w_HII=20, w_DIG=30):
#    fig = plt.figure(figsize=(4,4))
#    fig.canvas.set_window_title('Canvas active title')
#    ax = fig.add_subplot(111)
    
    (ny,nx) = Ha_image_clean.shape
    image_w = np.ones((ny,nx))
    dr = FWHM_MUSE
    for blob in blobs:
        y, x, r = blob
        i0=int(x-dr*r)
        i1=int(x+dr*r)
        j0=int(y-dr*r)
        j1=int(y+dr*r)
        if (i0<0):
            i0=0
        if (j0<0):
            j0=0
        if (i1>(nx-1)):
            i1=nx-1
        if (j1>(ny-1)):
            j1=ny-1
        image_sec=image_w[j0:j1,i0:i1] 
        (ny_sec,nx_sec)=image_sec.shape
        x_g = np.arange(0,nx_sec,1)
        y_g = np.arange(0,ny_sec,1)        
        x_g, y_g = np.meshgrid(x_g, y_g) # get 2D variables instead of 1D
        xp_g = x-i0
        yp_g = y-j0
        w_g = gaus2d(x_g, y_g, mx=xp_g, my=yp_g,sx=r, sy=r)
        image_w_now = 1.0+w_HII*(w_g)
        image_w[j0:j1,i0:i1]=image_w[j0:j1,i0:i1]/image_w_now
    for blob in diff_points:
        y, x = blob
        r = 3
        i0=int(x-dr*r)
        i1=int(x+dr*r)
        j0=int(y-dr*r)
        j1=int(y+dr*r)
        if (i0<0):
            i0=0
        if (j0<0):
            j0=0
        if (i1>(nx-1)):
            i1=nx-1
        if (j1>(ny-1)):
            j1=ny-1
        image_sec=image_w[j0:j1,i0:i1] 
        (ny_sec,nx_sec)=image_sec.shape
        x_g = np.arange(0,nx_sec,1)
        y_g = np.arange(0,ny_sec,1)        
        x_g, y_g = np.meshgrid(x_g, y_g) # get 2D variables instead of 1D
        xp_g = x-i0
        yp_g = y-j0
        w_g = gaus2d(x_g, y_g, mx=xp_g, my=yp_g,sx=r, sy=r)
        image_w_now = 1.0+w_DIG*(w_g)
        image_w[j0:j1,i0:i1]=image_w[j0:j1,i0:i1]*image_w_now

    image_diff = np.zeros((ny,nx))
    image_Ha = Ha_image_clean*image_w
    r_lim = 3*FWHM_MUSE
#
# No weight!
#
    r = 3
    for j in range(ny):
        for i in range(nx):
            i0=int(i-dr*r)
            i1=int(i+dr*r)
            j0=int(j-dr*r)
            j1=int(j+dr*r)
            if (i0<0):
                i0=0
            if (j0<0):
                j0=0
            if (i1>(nx-1)):
                i1=nx-1
            if (j1>(ny-1)):
                j1=ny-1
            image_sec=image_Ha[j0:j1,i0:i1]             
            image_w_sec=image_w[j0:j1,i0:i1] 
            (ny_sec,nx_sec)=image_sec.shape
            x_g = np.arange(0,nx_sec,1)
            y_g = np.arange(0,ny_sec,1)
            x_g, y_g = np.meshgrid(x_g, y_g) # get 2D variables instead of 1D
            xp_g = i-i0
            yp_g = j-j0
            dist = np.sqrt((x_g-xp_g)**2+(y_g-yp_g)**2)
#            image_sec[dist>2*dr] = 0
#            image_w_sec[dist>2*dr] = 0            
#            print(dist)
            val1 = np.nansum(np.ma.masked_array(image_sec,\
                                                mask=dist>3.5*r))#np.nansum(image_sec)
            val2 = np.nansum(np.ma.masked_array(image_w_sec,\
                                                mask=dist>3.5*r))#np.nansum(image_sec)
#            val2 = np.ma.sum(image_w_sec, mask=dist>3*r)#np.nansum(image_sec)
#            val2 = np.nansum(image_w_sec)
            val = val1/val2 #np.nansum(image_sec)/np.nansum(image_w_sec)
            image_diff[j,i] = val
#            ax.cla()
#            ax.imshow(np.ma.masked_array(dist,mask=dist>2*r))
#            ax.imshow(np.ma.masked_array(image_w_sec,mask=dist>2*r))
#            ax.set_title('[i,j]='+str(i)+','+str(j)+' = '+str(val1)+\
#                         ' '+str(val2)+' '+str(val))
#            fig.canvas.draw()

        #print(val)
#        print(j)
    image_diff=gaussian_filter(image_diff, sigma=1.5)
    return image_diff

def create_diff_cube(cube_clean,blobs,FWHM_MUSE,diff_points, w_HII=20, w_DIG=30):
#    fig = plt.figure(figsize=(4,4))
#    fig.canvas.set_window_title('Canvas active title')
#    ax = fig.add_subplot(111)
    
    (nz,ny,nx) = cube_clean.shape
    #
    # Creating weight image
    #
    image_w = np.ones((ny,nx))
    dr = FWHM_MUSE
    for blob in blobs:
        y, x, r = blob
        i0=int(x-dr*r)
        i1=int(x+dr*r)
        j0=int(y-dr*r)
        j1=int(y+dr*r)
        if (i0<0):
            i0=0
        if (j0<0):
            j0=0
        if (i1>(nx-1)):
            i1=nx-1
        if (j1>(ny-1)):
            j1=ny-1
        image_sec=image_w[j0:j1,i0:i1] 
        (ny_sec,nx_sec)=image_sec.shape
        x_g = np.arange(0,nx_sec,1)
        y_g = np.arange(0,ny_sec,1)        
        x_g, y_g = np.meshgrid(x_g, y_g) # get 2D variables instead of 1D
        xp_g = x-i0
        yp_g = y-j0
        w_g = gaus2d(x_g, y_g, mx=xp_g, my=yp_g,sx=r, sy=r)
        image_w_now = 1.0+w_HII*(w_g)
        image_w[j0:j1,i0:i1]=image_w[j0:j1,i0:i1]/image_w_now
    for blob in diff_points:
        y, x = blob
        r = 3
        i0=int(x-dr*r)
        i1=int(x+dr*r)
        j0=int(y-dr*r)
        j1=int(y+dr*r)
        if (i0<0):
            i0=0
        if (j0<0):
            j0=0
        if (i1>(nx-1)):
            i1=nx-1
        if (j1>(ny-1)):
            j1=ny-1
        image_sec=image_w[j0:j1,i0:i1] 
        (ny_sec,nx_sec)=image_sec.shape
        x_g = np.arange(0,nx_sec,1)
        y_g = np.arange(0,ny_sec,1)        
        x_g, y_g = np.meshgrid(x_g, y_g) # get 2D variables instead of 1D
        xp_g = x-i0
        yp_g = y-j0
        w_g = gaus2d(x_g, y_g, mx=xp_g, my=yp_g,sx=r, sy=r)
        image_w_now = 1.0+w_DIG*(w_g)
        image_w[j0:j1,i0:i1]=image_w[j0:j1,i0:i1]*image_w_now

    #
    # Extracting weighted cube
    #
    cube_diff = np.zeros((nz,ny,nx))
    cube_input = cube_clean*image_w
    r_lim = 3*FWHM_MUSE
#
# No weight!
#
    r = 3
    for j in range(ny):
        for i in range(nx):
            i0=int(i-dr*r)
            i1=int(i+dr*r)
            j0=int(j-dr*r)
            j1=int(j+dr*r)
            if (i0<0):
                i0=0
            if (j0<0):
                j0=0
            if (i1>(nx-1)):
                i1=nx-1
            if (j1>(ny-1)):
                j1=ny-1
            image_sec=cube_input[:,j0:j1,i0:i1]             
            image_w_sec=image_w[j0:j1,i0:i1] 
            (ny_sec,nx_sec)=image_w_sec.shape
            x_g = np.arange(0,nx_sec,1)
            y_g = np.arange(0,ny_sec,1)
            x_g, y_g = np.meshgrid(x_g, y_g) # get 2D variables instead of 1D
            xp_g = i-i0
            yp_g = j-j0
            dist = np.sqrt((x_g-xp_g)**2+(y_g-yp_g)**2)
            dist_cube = np.zeros((nz,ny_sec,nx_sec))
            for indx,img_dist in enumerate(dist_cube):
                dist_cube[indx,:,:]=dist
#            image_sec[dist>2*dr] = 0
#            image_w_sec[dist>2*dr] = 0            
#            print(dist)
            masked_vals = np.ma.masked_array(image_sec,\
                                             mask=dist_cube>3.5*r)
            val1 = masked_vals.sum(axis=1).sum(axis=1)
            val2 = np.nansum(np.ma.masked_array(image_w_sec,\
                                                mask=dist>3.5*r))#np.nansum(image_sec)
#            val2 = np.ma.sum(image_w_sec, mask=dist>3*r)#np.nansum(image_sec)
#            val2 = np.nansum(image_w_sec)
            val = val1/val2 #np.nansum(image_sec)/np.nansum(image_w_sec)
            cube_diff[:,j,i] = val
#            ax.cla()
#            ax.imshow(np.ma.masked_array(dist,mask=dist>2*r))
#            ax.imshow(np.ma.masked_array(image_w_sec,mask=dist>2*r))
#            ax.set_title('[i,j]='+str(i)+','+str(j)+' = '+str(val1)+\
#                         ' '+str(val2)+' '+str(val))
#            fig.canvas.draw()

        #print(val)
#        print(j)
#    i=0
    for i,img_diff in enumerate(cube_diff):
        cube_diff[i,:,:]=gaussian_filter(img_diff, sigma=1.5)
        i=i+1
    return cube_diff



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
