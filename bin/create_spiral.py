#!/usr/bin/python3
import numpy as np
import scipy as sp
from astropy.io import fits
import matplotlib.pyplot as plt
import PIL
from PIL import Image,ImageEnhance
import matplotlib.image as mpimg
import matplotlib.colors as colors
from math import sqrt
import sys
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from matplotlib.colors import LogNorm
from astropy.io import ascii
from scipy.ndimage import gaussian_filter, correlate
from scipy.spatial import Delaunay, cKDTree, KDTree
from scipy.interpolate import griddata
import matplotlib.image as mpimg
from scipy.optimize import minimize, curve_fit, leastsq
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.colors import LinearSegmentedColormap
import math
import img_scale
from scipy import ndimage
import os
import pandas as pd
import warnings
import pandas as pd
from astropy.table import QTable, Table, Column
from astropy import units as u 
from pyHIIExplorer.HIIblob import *
from pyHIIExplorer.extract import *
import time
import argparse
warnings.filterwarnings('ignore')
from matplotlib import rcParams as rc


#Create spiral
def create_spiral(r=0.85,ab=0.85,PA=45, A=0.5,B=1,C=5,N_sp=2):
    ang = np.linspace(-3.1416,3.1416,100)
    x_circ=r*np.sin(ang)+r*np.cos(ang)
    y_circ=-r*np.cos(ang)+r*np.sin(ang)
    pa=(PA/180)*3.1416
    ang = np.linspace(-3.1416,3.1416,100)
    x_e_o=r*np.sin(ang)+r*np.cos(ang) 
    y_e_o=ab*(-r*np.cos(ang)+r*np.sin(ang))
    x_e = x_e_o*np.cos(pa)+y_e_o*np.sin(pa)
    y_e = x_e_o*np.sin(pa)-y_e_o*np.cos(pa)
    
    ang = np.linspace(0,2*3.1416,100)
    R_sp=A/np.log(B*np.tan(ang/(2*C))) #equation 1 SFS 2012
    n_el=ang.shape[0]
    x_e_sp_o=np.zeros(n_el*N_sp)
    y_e_sp_o=np.zeros(n_el*N_sp)
    
    for i_sp in range(N_sp):
        n_sp1=i_sp*n_el
        n_sp2=(i_sp+1)*n_el
        x_e_sp_o[n_sp1:n_sp2]  = R_sp*np.cos(ang+i_sp*3.1416/N_sp*2)
        y_e_sp_o[n_sp1:n_sp2] = ab*R_sp*np.sin(ang+i_sp*3.1416/N_sp*2)

    r_dist = A*np.sqrt(x_e_sp_o**2+y_e_sp_o**2)
    x_e_sp = x_e_sp_o*np.cos(pa)+y_e_sp_o*np.sin(pa)
    y_e_sp = x_e_sp_o*np.sin(pa)-y_e_sp_o*np.cos(pa)
    np.random.seed(0)
    x_e_sp_r = x_e_sp+A*0.7*(np.random.rand(*x_e_sp_o.shape)-0.5)
    y_e_sp_r = y_e_sp+A*0.7*(np.random.rand(*x_e_sp_o.shape)-0.5)
    d_r=np.sqrt((x_e_sp_r-x_e_sp)**2+(y_e_sp_r-y_e_sp)**2)
    mask_sp = np.isfinite(x_e_sp) & np.isfinite(y_e_sp)# & (x_e_sp>0)
    x_e_sp=x_e_sp[mask_sp]
    y_e_sp=y_e_sp[mask_sp]
    r_e_sp=np.sqrt(x_e_sp**2+y_e_sp**2)
    np.random.seed(0)
    N2_sp=-0.5-0.1*r_e_sp*2+0.4*(np.random.rand(*x_e_sp_o.shape)-0.5)
    O3_sp=0.5/(N2_sp+0.1)+0.5+0.4*(np.random.rand(*x_e_sp_o.shape)-0.5)
    EW_sp = 0.5+0.95+0.51*r_e_sp*2
    
    return x_e_sp,y_e_sp,r_e_sp,x_e_sp_r,y_e_sp_r,N2_sp,O3_sp,EW_sp,r_dist


#Leaking of HII regions (flow drop)
def r_2(x=0, y=0, mx=0, my=0, r_h=1): 
    r_0=np.sqrt((x - mx)**2.+ (y - my)**2.)
    r = r_0+r_h
    w=r_h**2/r**2
    return np.ma.masked_array(w,mask=r_0<0.5*r_h, fill_value=0.44)


#Create leaking
def create_HII_image_leaking(blobs,flux,nx,ny,dr=5, fleak=0.1):
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
        image_sec=image[j0:j1,i0:i1] 
        (ny_sec,nx_sec)=image_sec.shape
        x_g = np.arange(0,nx_sec,1)
        y_g = np.arange(0,ny_sec,1)        
        x_g, y_g = np.meshgrid(x_g, y_g) # get 2D variables instead of 1D
        xp_g = x-i0
        yp_g = y-j0
        w_g = r_2(x_g, y_g, mx=xp_g, my=yp_g, r_h=r)
        image_HII = bflux*w_g*fleak
        if ((~np.isnan(bflux)) and (bflux>0)):
            image[j0:j1,i0:i1]=image[j0:j1,i0:i1]+image_HII
    return image


#Create DIG
def create_DIG (x_c,y_c,PA,ab,h_scale,cont_peak,size):
    nx_in = 2*x_c
    ny_in = 2*y_c 
    x = np.arange(0, nx_in, 1)
    y = np.arange(0, ny_in, 1)
    xv, yv = np.meshgrid(x, y)
    pa=(PA/180)*3.1416
    x_e_sp = ab*((xv-x_c)*np.cos(pa)+(yv-y_c)*np.sin(pa))
    y_e_sp = ((xv-x_c)*np.sin(pa)-(yv-y_c)*np.cos(pa))
    rv = np.sqrt(x_e_sp**2+y_e_sp**2)
    rv_disk = rv/h_scale
    cont_v = cont_peak*np.exp(-rv_disk)
    EW_v = 0.5+1.0*np.random.rand(ny_in,nx_in)
    Ha_v = EW_v*cont_v
    slope_rand = 0.25+0.2*np.random.normal(0,0.3,(ny_in,nx_in))
    N2_v = 0.2+slope_rand[0]*rv_disk/size-0.25*(np.random.normal(0,0.3,(ny_in,nx_in))-0.5)
    slope_rand = 1.7/4+1.3*(np.random.normal(0,0.3,(ny_in,nx_in))-0.5)
    O3_v=0.15+slope_rand[0]*rv_disk/size-0.25*(np.random.normal(0,0.3,(ny_in,nx_in))-0.5)
    NII_v = 10**(N2_v)*Ha_v
    Hb_v = Ha_v/2.86
    OIII_v = 10**(O3_v)*Hb_v
    cube_v=np.array((Hb_v,OIII_v,Ha_v,NII_v, EW_v))
    return cube_v
    

#Create BPT type Seaborn (single)
def create_bpt_sns_single (x_plt, y_plt, cmap, c, x_min1,x_max1, y_min1, y_max1, legend, namesave):  #counts
    x=np.linspace(x_min1,x_max1,500)
    cut_y=-0.7+0.2-3.67*x
    cut_y2=-1.7+0.5-3.67*x
    cut_y3=0.61/(x-0.05)+1.3
    cut_y4=0.61/(x-0.47)+1.19
    cut_y5_ce=0.13/(x-0.003)+0.57
    cut_y_SII=0.72/(x-0.32)+1.3;
    cut_y_SII_ce=0.04/(x+0.012)+0.58
    cut_y_SII_AGNs=1.89*(x)+0.76;
    cut_y_OI=0.73/((x+0.59))+1.33;#+1.10;
    cut_y_OI_AGNs=1.18*(x)+1.30;
    cut_y_OI_ce=0.056/(x+0.40)+0.61
    cut_y_OIII_AGNs=1.14*(x)+0.36;
    
    fig = plt.figure()
    maska1 = (x_plt>=-5000)&(x_plt<=5000)& (y_plt>=-5000)&(y_plt<=5000)
    x_plt = x_plt[maska1]
    y_plt = y_plt[maska1]
    df = pd.DataFrame({"x":x_plt,"y":y_plt})
    ax=sns.jointplot(data=df, x='x', y='y', kind="kde", cmap=cmap, vmin=0,vmax=20, xlim=[x_min1+2,x_max1+0.1], ylim=[y_min1,y_max1],shade=False, 
                   shade_lowest=False, alpha=1, marginal_kws={"color": c, "alpha": 0.9}, levels=[0.15, 0.5, 0.99])
                   
                   #joint_kws={'weights':counts})
    
    ax.ax_joint.plot(x[x<-0.01], cut_y3[x<-0.01], '#D6A02B', linestyle='--', linewidth=3)
    ax.ax_joint.plot(x,cut_y4, c='#006476',linestyle='-.',linewidth=3)
    ax.ax_joint.plot(x[x<-0.01],cut_y5_ce[x<-0.01],c="#3b001d", linestyle=':',linewidth=3)
    
    ax.ax_joint.legend_.remove()
    ax.set_axis_labels(r'log([NII]/H$\alpha$)',r'log([OIII]/H$\beta$)', size=28)

    plt.subplots_adjust(left=0.15, right=1, top=1, bottom=0.15, wspace=0, hspace=0)
    plt.legend([legend],  loc='3',bbox_to_anchor=(-1.2, 0.3), prop={'size': 22})
    plt.savefig(namesave+".png")
    plt.show()
    return fig


#Create BPT type Seaborn (double)
def create_bpt_sns_double (x_plt1, y_plt1, x_plt2, y_plt2, cmap1, cmap2, c1, c2, x_min1, x_max1, y_min1, y_max1, legend, namesave):
    x=np.linspace(x_min1,x_max1,500)
    cut_y=-0.7+0.2-3.67*x
    cut_y2=-1.7+0.5-3.67*x
    cut_y3=0.61/(x-0.05)+1.3
    cut_y4=0.61/(x-0.47)+1.19
    cut_y5_ce=0.13/(x-0.003)+0.57
    cut_y_SII=0.72/(x-0.32)+1.3;
    cut_y_SII_ce=0.04/(x+0.012)+0.58
    cut_y_SII_AGNs=1.89*(x)+0.76;
    cut_y_OI=0.73/((x+0.59))+1.33;#+1.10;
    cut_y_OI_AGNs=1.18*(x)+1.30;
    cut_y_OI_ce=0.056/(x+0.40)+0.61
    cut_y_OIII_AGNs=1.14*(x)+0.36;

    fig = plt.figure()
    maska2 = (x_plt2>=-5000)&(x_plt2<=5000)&(y_plt2>=-5000)&(y_plt2<=5000)
    maska1 = (x_plt1>=-5000)&(x_plt1<=5000)&(y_plt1>=-5000)&(y_plt1<=5000)
    x_plt2 = x_plt2[maska2]
    y_plt2 = y_plt2[maska2]
    x_plt1 = x_plt1[maska1]
    y_plt1 = y_plt1[maska1]
    df = pd.DataFrame({"x":x_plt1,"y":y_plt1})
    ax=sns.jointplot(data=df, x='x', y='y', kind="kde", cmap=cmap1, vmin=0, vmax=20, xlim=[x_min1+2,x_max1+0.1], ylim=[y_min1,y_max1], shade=False, 
                   shade_lowest=False, alpha=1, marginal_kws={"color": c1, "alpha": 0.5}, levels=[0.15, 0.5, 0.99])
    ax.x = x_plt2
    ax.y = y_plt2
    ax.plot_joint(sns.kdeplot, cmap=cmap2, vmin=0, vmax=20, shade=False, shade_lowest=False, alpha=1 , levels=[0.15, 0.5, 0.99])

    ax.ax_joint.plot(x[x<-0.01], cut_y3[x<-0.01], '#D6A02B', linestyle='--', linewidth=3)
    ax.ax_joint.plot(x,cut_y4, c='#006476',linestyle='-.',linewidth=3)
    ax.ax_joint.plot(x[x<-0.01],cut_y5_ce[x<-0.01],c="#3b001d", linestyle=':',linewidth=3)

    ax.ax_joint.legend_.remove()
    ax.set_axis_labels(r'log([NII]/H$\alpha$)',r'log([OIII]/H$\beta$)', size=33)
    ax.plot_marginals(sns.kdeplot, color=c2, shade=True, alpha=1)

    plt.subplots_adjust(left=0.17, right=1, top=1, bottom=0.15, wspace=0, hspace=0)
    plt.legend(legend,  loc='3', bbox_to_anchor=(-0.8, 0.4), prop={'size': 30})
    plt.savefig(namesave+".png")
    plt.show()
    return fig

def create_cont (ny_in, nx_in, h_scale, cont_peak, x_c, y_c, PA, ab):
    pa = (PA/180)*3.1416
    img_cont = np.zeros((ny_in,nx_in))
    for j in np.arange(0, ny_in):
        for i in np.arange(0, nx_in):
            x_now = i-x_c
            y_now = j-y_c
            y_e = x_now*np.cos(pa)+y_now*np.sin(pa)
            x_e = x_now*np.sin(pa)-y_now*np.cos(pa)
            r_dist_now = np.sqrt((y_e*ab)**2+x_e**2)/h_scale
            img_cont[j,i] = cont_peak*np.exp(-r_dist_now)
    return img_cont
    
    
def create_whan(x_plt1, y_plt1, x_plt2, y_plt2, cmap1, cmap2, c1, c2, x_min, x_max, y_min, y_max,  name, DIR):
    
    fig = plt.figure()
    
    maska2 = (x_plt2>=-5000)&(x_plt2<=5000)&(y_plt2>=-5000)&(y_plt2<=5000)
    maska1 = (x_plt1>=-5000)&(x_plt1<=5000)&(y_plt1>=-5000)&(y_plt1<=5000)
    x_plt2 = x_plt2[maska2]
    y_plt2 = y_plt2[maska2]
    x_plt1 = x_plt1[maska1]
    y_plt1 = y_plt1[maska1]
    df = pd.DataFrame({"x":x_plt1,"y":y_plt1})
    ax=sns.jointplot(data=df, x='x', y='y', kind="kde", cmap=cmap1, vmin=0, vmax=20, xlim=[x_min, x_max], ylim=[y_min, y_max], shade=False, 
                   shade_lowest=False, alpha=0.5, marginal_kws={"color": c1, "alpha": 0.5}, levels=[0.15, 0.5, 0.99])
    
    ax.x = x_plt2
    ax.y = y_plt2
    ax.plot_joint(sns.kdeplot, cmap=cmap2, vmin=0, vmax=20, shade=False, shade_lowest=False, alpha=1 , levels=[0.15, 0.5, 0.99])
    ax.ax_joint.legend_.remove()
    ax.set_axis_labels(r'log([NII]/H$\alpha$)', r'log W$_{H\alpha}$ ($\AA$)', size=32)
    ax.plot_marginals(sns.kdeplot, color=c2, shade=True, alpha=1)

    ax.ax_joint.plot((-0.14,-0.14), (0.5,y_max) ,c='#646464', ls="--", linewidth=2)
    ax.ax_joint.plot((-0.14,x_max), (0.79,0.79), c='#646464', ls="--", linewidth=2)
    ax.ax_joint.plot((x_min,x_max), (0.5,0.5), c='#646464', ls="--", linewidth=2)


    plt.text(0.25, 1.0, 'SF', fontsize=38, fontweight=800,transform=fig.transFigure)
    plt.text(0.6, 1.0, 'sAGN', fontsize=38, fontweight=800, transform=fig.transFigure)
    plt.text(0.6, 0.6, 'wAGN', fontsize=38, fontweight=800, transform=fig.transFigure)
    plt.text(0.25, 0.3, 'Retired', fontsize=38, fontweight=800, transform=fig.transFigure)

    plt.subplots_adjust(left=0.2, right=1, top=1, bottom=0.15, wspace=0, hspace=0)
    plt.savefig(DIR+'/'+name)
    plt.show()
    return fig
    
def radi(x_plt1, y_plt1, x_plt2, y_plt2, cmap1, cmap2, c1, c2, x_min, x_max, y_min, y_max, legend, name, DIR):

    fig = plt.figure()
    maska2 = (x_plt2>=-5000)&(x_plt2<=5000)&(y_plt2>=-5000)&(y_plt2<=5000)
    maska1 = (x_plt1>=-5000)&(x_plt1<=5000)&(y_plt1>=-5000)&(y_plt1<=5000)
    x_plt2 = x_plt2[maska2]
    y_plt2 = y_plt2[maska2]
    x_plt1 = x_plt1[maska1]
    y_plt1 = y_plt1[maska1]
    df = pd.DataFrame({"x":x_plt1,"y":y_plt1})
    ax=sns.jointplot(data=df, x='x', y='y', kind="kde", cmap=cmap1, vmin=0, vmax=20, xlim=[x_min, x_max], ylim=[y_min, y_max], shade=False, 
                   shade_lowest=False, alpha=1, marginal_kws={"color": c1, "alpha": 0.5}, levels=[0.15, 0.5, 0.99])
    ax.x = x_plt2
    ax.y = y_plt2
    ax.plot_joint(sns.kdeplot, cmap=cmap2, vmin=0, vmax=20, shade=False, shade_lowest=False, alpha=1, levels=[0.15, 0.5, 0.99])

    ax.ax_joint.legend_.remove()
    ax.set_axis_labels(r'log(flux H$\alpha$)', r'log(radius)', size=32)
    ax.plot_marginals(sns.kdeplot, color=c2, shade=True, alpha=0.5)

    plt.subplots_adjust(left=0.17, right=0.98, top=1, bottom=0.15, wspace=0, hspace=0)
    #plt.legend(legend, bbox_to_anchor=(-1.7, 1), prop={'size': 28})  #
    #plt.tight_layout()
    plt.savefig(DIR+'/'+name+".png")
    plt.show()
    return fig


#Create simulations

def spiral_sim(name, r, PA, ab, A, B, C, N_sp, x_c, y_c, size, FWHM, spax_scale, cont_peak, f_leak, max_size, plot, DIR):


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
    #
    # Recalculate the EW_sp
    #
    EW_scale = np.sqrt(2*np.pi)*np.mean(rad_HII_arc)
    EW_sp = EW_sp - np.log10(np.sqrt(2*np.pi)*rad_HII_arc)+np.log10(EW_scale)

    R_in = rad_HII_arc/np.sqrt(2)
    X_in = x_e_sp_r
    Y_in = y_e_sp_r
    flux_in  = Ha
    blobs_in = np.swapaxes(np.array((Y_in,X_in,R_in)),axis1=0,axis2=1)
    
    ####Create cont####
    img_cont = create_cont(ny_in=ny_in, nx_in=nx_in, h_scale=h_scale, cont_peak=cont_peak, x_c=x_c, y_c=y_c, PA=PA, ab=ab)
    img_cont = img_cont/EW_scale
    #/(np.mean(r_sp)*2.5)**2
    #/(spax_scale/FWHM)**2
    
    #img_cont = cont/np.sqrt(2*np.pi)*rad_HII_Arc
    
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
    
    #if (plot==1):
    #    plt.imshow(img_Ha_leak)
    #    plt.xlim(0,nx_in)
    #    plt.ylim(0,ny_in)
    #    plt.savefig(DIR+'/'+name+'_leak.png')
    #    plt.show()
        
    ####Create holmes####
    cube_holmes = create_DIG (x_c, y_c , PA, ab, h_scale, cont_peak, size)

    f_scale = 2
    hdu_cube_holmes = fits.PrimaryHDU(data = cube_holmes*f_scale)
    hdu_cube_holmes.writeto(DIR+'/'+'holmes.cube.fits.gz', overwrite=True)
    
    #if (plot==1):
    #    plt.imshow(cube_holmes[2,:,:])
    #    plt.xlim(0,nx_in)
    #    plt.ylim(0,ny_in)
    #    plt.savefig(DIR+'/'+name+'_holmes.png')
    #    plt.show()

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
    
    #if (plot==1):
    #    plt.imshow(img_Ha)
    #    plt.xlim(0,nx_in)
    #    plt.ylim(0,ny_in)
    #    plt.savefig(DIR+'/'+name+'_onlyHIIregions.png')
    #    plt.show()
    
    halfaleaking = np.sum(img_Ha_leak)
    halfahiiregions = np.sum(img_Ha)
    percentage_fleak = halfaleaking/halfahiiregions
    
    
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
    
    file_sim = DIR+'/'+'sim.cube.fits.gz'
    file_sim = fits.open(file_sim)
    cube_simulate = file_sim[0].data
    
    Ha_sim=cube_simulate[2,:,:]
    Hb_sim=cube_simulate[0,:,:] 
    OIII_sim=cube_simulate[1,:,:]
    NII_sim=cube_simulate[3,:,:]
    
    O3_sim=np.log10(OIII_sim)-np.log10(Hb_sim)
    N2_sim=np.log10(NII_sim)-np.log10(Ha_sim)
    
    if (plot==1):
        #os.system('/usr/bin/python3 RGB_img_cube_sim.py 60_leak/sim.cube.fits.gz sim_test.png')
        sim_test = RGB_img_cube(fits_file=DIR+'/sim.cube.fits.gz', output_file=DIR+'/sim_test.png') 
        img_all = mpimg.imread(DIR+'/'+'sim_test.png')
        plt.imshow(img_all)
        plt.xlim(0,nx_in)
        plt.ylim(0,ny_in)
        plt.savefig(DIR+'/'+name+'_all_RGB.png')
        plt.show()
        
    img_Ha_ALL = cube_ALL[2,:,:]
    
    
    FWHM=FWHM/spax_scale 

    blobs_final,blobs_F_Ha,image_HII, diff_map_final,diff_points,diff_Flux = HIIblob(img_Ha_ALL,img_cont_Ha,FWHM, MUSE_1sig=1.0, MUSE_1sig_V=0.1,
    											plot=plot, refined=3, num_sigma=300, name=name, max_size=max_size, DIR=DIR)
                                                                                

    WCS, hdu_HII_out, hdu_DIG_out, table_HII, table_DIG = extracting_flux_elines('sim', hdu_cube_ALL, blobs_final, diff_points, FWHM,  plot=plot)

    cube_diff_output = create_diff_cube(hdu_cube_ALL.data-hdu_HII_out.data,blobs_final,FWHM,diff_points)
    
    #cube_diff_output = create_diff_cube(hdu_DIG_out.data,blobs_final,FWHM,diff_points) difuso feo
    
    cube_diff_output[4,:,:] = (cube_diff_output[2,:,:]/hdu_cube_ALL.data[2,:,:])*hdu_cube_ALL.data[4,:,:] #modificando EW DIG
    
    hdu_HII_out.data[4,:,:] = (hdu_HII_out.data[2,:,:]/hdu_cube_ALL.data[2,:,:])*hdu_cube_ALL.data[4,:,:] #modificando EW HII

    hdu_DIG_out.data = cube_diff_output
    
    hdu_HII_out.data = hdu_HII_out.data
    
    hdu_SUM_out = hdu_HII_out.copy()
    hdu_RES_out = hdu_HII_out.copy()
    hdu_SUM_out.data = hdu_SUM_out.data+hdu_DIG_out.data
    hdu_RES_out.data = hdu_cube_ALL.data-(hdu_SUM_out.data)
    
    
    hdu_HII_out.writeto(DIR+'/'+'HII_out.cube.fits.gz', overwrite=True)
    hdu_DIG_out.writeto(DIR+'/'+'DIG_out.cube.fits.gz', overwrite=True)
    hdu_SUM_out.writeto(DIR+'/'+'SUM_out.cube.fits.gz', overwrite=True)
    hdu_RES_out.writeto(DIR+'/'+'RES_out.cube.fits.gz', overwrite=True)
    
    
    file_sum = DIR+'/'+'SUM_out.cube.fits.gz'
    file_sum = fits.open(file_sum)
    cube_sum = file_sum[0].data
    
    Ha_sum=cube_sum[2,:,:]
    Hb_sum=cube_sum[0,:,:] 
    OIII_sum=cube_sum[1,:,:]
    NII_sum=cube_sum[3,:,:]
    
    O3_sum=np.log10(OIII_sum)-np.log10(Hb_sum)
    N2_sum=np.log10(NII_sum)-np.log10(Ha_sum)
    
    
    #os.system('/usr/bin/python3 RGB_img_cube_sim.py 60_leak/HII_out.cube.fits.gz 60_leak/HII_out.png')
    #os.system('/usr/bin/python3 RGB_img_cube_sim.py 60_leak/HIIregions.cube.fits.gz 60_leak/HII_inp.png')
    #os.system('/usr/bin/python3 RGB_img_cube_sim.py 60_leak/DIG_out.cube.fits.gz 60_leak/DIG_out.png')
    #os.system('/usr/bin/python3 RGB_img_cube_sim.py 60_leak/diff_all.cube.fits.gz 60_leak/DIG_inp.png')
    #os.system('/usr/bin/python3 RGB_img_cube_sim.py 60_leak/SUM_out.cube.fits.gz 60_leak/SUM_out.png')
    #os.system('/usr/bin/python3 RGB_img_cube_sim.py 60_leak/sim.cube.fits.gz 60_leak/SIM_inp.png')

    if (plot==1):
        
        HII_out_img = RGB_img_cube(fits_file=DIR+'/HII_out.cube.fits.gz', output_file=DIR+'/HII_out.png')
        HII_inp_img = RGB_img_cube(fits_file=DIR+'/HIIregions.cube.fits.gz', output_file=DIR+'/HII_inp.png')
        DIG_out_img = RGB_img_cube(fits_file=DIR+'/DIG_out.cube.fits.gz', output_file=DIR+'/DIG_out.png')
        diff_all_img = RGB_img_cube(fits_file=DIR+'/diff_all.cube.fits.gz', output_file=DIR+'/DIG_inp.png')
        SUM_out_img = RGB_img_cube(fits_file=DIR+'/SUM_out.cube.fits.gz', output_file=DIR+'/SUM_out.png')
        SIM_inp_img = RGB_img_cube(fits_file=DIR+'/sim.cube.fits.gz', output_file=DIR+'/SIM_inp.png')

    
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

    if(plot==1):

        x_min=-1.3
        x_max=1.1
        y_min=-1.3
        y_max=4
    
        whan_digregions = create_whan(x_plt1=N2_DIG_inp, y_plt1=ew_DIG_inp, x_plt2=N2_DIG_out, y_plt2=ew_DIG_out, cmap1='Wistia_r', cmap2='seismic_r',
                                      c1='#ff7214', c2='#a80000', x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, name=name+'__WHAN_DIG.png', DIR=DIR)
    
        whan_hiiregions = create_whan(x_plt1=N2_HII_inp, y_plt1=ew_HII_inp, x_plt2=N2_HII_out, y_plt2=ew_HII_out, cmap1='winter_r', cmap2='PiYG_r',
                                      c1='#7bffa7', c2='#008f39', x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, name=name+'__WHAN_HII.png', DIR=DIR)


        x_min1=-4
        x_max1=0.45
        y_min1=-2.5
        y_max1=1.2
        
        bpt_hiiregions = create_bpt_sns_double(x_plt1=N2_HII_inp, y_plt1=O3_HII_inp, x_plt2=N2_HII_out, y_plt2=O3_HII_out, cmap1='winter_r', cmap2='PiYG_r',
                                               c1='#7bffa7', c2='#008f39', x_min1=x_min1, x_max1=x_max1,
                                               y_min1=y_min1,y_max1=y_max1, legend=['Input HII', 'Output HII'], namesave=DIR+'/'+name+'bpt_hiiregions')

        bpt_dig = create_bpt_sns_double(x_plt1=N2_DIG_inp, y_plt1=O3_DIG_inp, x_plt2=N2_DIG_out, y_plt2=O3_DIG_out, cmap1='Wistia_r', cmap2='seismic_r',
                                        c1='#ff7214', c2='#a80000', x_min1=x_min1, x_max1=x_max1,
                                        y_min1=y_min1,y_max1=y_max1, legend=['Input DIG', 'Output DIG'], namesave=DIR+'/'+name+'bpt_digregions')


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
        plt.savefig(DIR+'/'+name+'all.png')
        plt.show()
    
    
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
    Ha_HII_out = np.array(Ha_HII_out.flat)
    ew_HII_out = np.array(ew_HII_out.flat)
    
    
    #rat_Ha = (Ha_HII_out-Ha_HII_inp)/Ha_HII_inp
    #plt.imshow(rat_Ha, vmin=-2, vmax=2)
    #plt.show()
    #plt.imshow(Ha_HII_inp,vmin=-5, vmax=40)
    #plt.show()
    #plt.imshow(Ha_HII_out, vmin=-5, vmax=40)
    #plt.show()
    #print(rat_Ha)
    #print(np.nanmedian(rat_Ha))
    #print(np.nanmedian(Ha_HII_inp[Ha_HII_inp>0]))
    #print(np.nanmedian(Ha_HII_out[Ha_HII_out>0]))
    #print(np.nanstd(rat_Ha[Ha_HII_inp>0.5*np.nanmedian(Ha_HII_inp[Ha_HII_inp>0])]))
    #print(np.nanmean(rat_Ha[Ha_HII_inp>0.5*np.nanmedian(Ha_HII_inp[Ha_HII_inp>0])]))
   
    
    ##################################

    ew_DIG_inp = np.array((cube_DIG_inp[4,:,:]).flat)
    Ha_DIG_inp = np.array(cube_DIG_inp[2,:,:].flat)
    
    ew_DIG_out = np.array((cube_DIG_out[4,:,:]).flat)
    Ha_DIG_out = np.array((cube_DIG_out[2,:,:]).flat)
    
    ##################################
    
    
    mask_flux_05 = (blobs_final>0.5*np.nanmedian(flux_in[flux_in>0]))
    mask_flux_05_in = (blobs_in>0.5*np.nanmedian(flux_in[flux_in>0]))
    
    mask_flux_rad05 = (flux_in>0.5*np.nanmedian(flux_in[flux_in>0]))
    mask_flux_bl05 = (blobs_F_Ha>0.5*np.nanmedian(flux_in[flux_in>0]))

    
    mask_N2_HII = (N2_HII_inp>-100000000)&(N2_HII_inp<100000000)&(N2_HII_out>-100000000)&(N2_HII_out<100000000)
    mask_O3_HII = (O3_HII_inp>-100000000)&(O3_HII_inp<100000000)&(O3_HII_out>-100000000)&(O3_HII_out<100000000)

    mask_N2_DIG = (N2_DIG_inp>-100000000)&(N2_DIG_inp<100000000)&(N2_DIG_out>-100000000)&(N2_DIG_out<100000000)
    mask_O3_DIG = (O3_DIG_inp>-100000000)&(O3_DIG_inp<100000000)&(O3_DIG_out>-100000000)&(O3_DIG_out<100000000)
    
    mask_ew_HII = (ew_HII_inp>0.5*np.nanmedian(ew_HII_inp))
    mask_ew_DIG = (ew_DIG_inp>0.5*np.nanmedian(ew_DIG_inp))

    if(plot==1):
        
        histogram=radi(x_plt1=np.log10(Ha), y_plt1=np.log10(R_in), x_plt2=np.log10(blobs_F_Ha*np.sqrt(2)), y_plt2=np.log10(blobs_final[:,2]), cmap1='Purples_r', cmap2='Blues_r',
        		c1='#4C0B5F', c2='#0000FF', x_min=0, x_max=4.5, y_min=-0.13, y_max=0.65, legend=['Input', 'Output'], name=name+'__histogram.png', DIR=DIR)
        
        #fig = plt.figure(figsize=(70,15))
        #sns.set(style="white",font_scale=7)

        #fig.add_subplot(1, 5, 1)
        #axha=sns.distplot(Ha, color='#FFA420')
        #sns.distplot(blobs_F_Ha, color='#AF2B1E', ax=axha)
        #axha.set_yscale('log')
        #axha.set_xscale('log')
        #axha.set_xlabel(r'flux H$\alpha$')
        #axha.set_ylabel(r'frequency')
        #axha.autoscale()

        #fig.add_subplot(1, 5, 2)
        #axra=sns.distplot(R_in, color='#FFA420')
        #sns.distplot(blobs_final[:,2], color='#AF2B1E', ax=axra)
        #axra.set_yscale('log')
        #axra.set_xlabel(r'radius')
        #axra.set_ylabel(r'frequency')
        #axra.autoscale()
        
        #fig.add_subplot(1, 5, 3)
        #plt.scatter(np.log10(Ha),np.log10(R_in), color='#FFA420', s=600, marker='*', alpha=0.4, edgecolor='none')
        #plt.scatter(np.log10(blobs_F_Ha*np.sqrt(2)),np.log10(blobs_final[:,2]), s=600, marker='*', color='#AF2B1E', alpha=0.4, edgecolor='none')
        #plt.xlabel(r'log(flux H$\alpha$)')
        #plt.ylabel(r'log(radius)')

        #fig.add_subplot(1, 5, 4)
        #plt.scatter(N2_sim, N2_sum, color='#5B3A29', s=300, marker='*')
        #plt.xlabel(r'Input log([NII]/H$\alpha$)')
        #plt.ylabel(r'Output log([NII]/H$\alpha$)')
        #plt.xlim(-1,1)
        #plt.ylim(-1,2.8)

        #fig.add_subplot(1, 5, 5)
        #plt.scatter(O3_sim, O3_sum, color='#2A6478', s=300, marker='*')
        #plt.xlabel(r'Input log([OIII]/H$\beta$)')
        #plt.ylabel(r'Output log([OIII]/H$\beta$)')
        #plt.xlim(-1,1)
        #plt.ylim(-1,2.8)
        
        #plt.subplots_adjust(left = 0.05, right=0.99, bottom = 0.17, top=0.92, wspace = 0.4,  hspace = 0.004)
        #plt.savefig(DIR+'/'+"histogramas"+str(ab)+".png")
        #plt.savefig(DIR+'/'+"histogramas"+str(ab)+".pdf")
        #plt.show()


    table_info=Table(rows=[(r, PA, ab, A, B, C, N_sp, x_c, y_c, size, FWHM, spax_scale, cont_peak, f_leak, max_size, np.around(percentage_fleak*100,2), len(blobs_in), len(blobs_final), np.around(np.float((100*len(blobs_final))/len(blobs_in)), 2), np.around(np.float((100*len(blobs_final[mask_flux_05]))/len(blobs_in[mask_flux_05_in])),2), np.float(np.nanmean(((Ha_HII_out-Ha_HII_inp)/Ha_HII_inp)[Ha_HII_inp>0.5*np.nanmedian(Ha_HII_inp[Ha_HII_inp>0])])), np.float(np.nanstd(((Ha_HII_out-Ha_HII_inp)/Ha_HII_inp)[Ha_HII_inp>0.5*np.nanmedian(Ha_HII_inp[Ha_HII_inp>0])])), np.float(np.nanmean(R_in[mask_flux_rad05])), np.float(np.nanstd(R_in[mask_flux_rad05])), np.float(np.nanmean(blobs_final[:,2][mask_flux_bl05])), np.float(np.nanstd(blobs_final[:,2][mask_flux_bl05])), np.float(np.nanmean(N2_HII_out[mask_N2_HII]-N2_HII_inp[mask_N2_HII])), np.float(np.nanstd(N2_HII_out[mask_N2_HII]-N2_HII_inp[mask_N2_HII])), np.float(np.nanmean(O3_HII_out[mask_O3_HII]-O3_HII_inp[mask_O3_HII])), np.float(np.nanstd(O3_HII_out[mask_O3_HII]-O3_HII_inp[mask_O3_HII])), np.float(np.nanmean(((ew_HII_out-ew_HII_inp)/ew_HII_inp)[mask_ew_HII])), np.float(np.nanstd(((ew_HII_out-ew_HII_inp)/ew_HII_inp)[mask_ew_HII])), np.float(np.nanmean(N2_DIG_out[mask_N2_DIG]-N2_DIG_inp[mask_N2_DIG])), np.float(np.nanstd(N2_DIG_out[mask_N2_DIG]-N2_DIG_inp[mask_N2_DIG])), np.float(np.nanmean(O3_DIG_out[mask_O3_DIG]-O3_DIG_inp[mask_O3_DIG])), np.float(np.nanstd(O3_DIG_out[mask_O3_DIG]-O3_DIG_inp[mask_O3_DIG])), np.float(np.nanmean(((ew_DIG_out-ew_DIG_inp)/ew_DIG_inp)[mask_ew_DIG])), np.float(np.nanstd(((ew_DIG_out-ew_DIG_inp)/ew_DIG_inp)[mask_ew_DIG]))  )], names=('r', 'PA', 'ab', 'A', 'B', 'C', 'arms', 'cen_x', 'cen_y', 'size', 'FWHM', 'spax_scale', 'cont_peak', 'f_leak','max_size','%f_leak', 'inp_hiiregions', 'out_hiiregions', '%_recovery', '%_recovery_05', 'delta_Ha_05', 'std_delta_Ha_05', 'mean_inp_R_05', 'std_inp_R_05', 'mean_out_R_05', 'std_out_R_05', 'mean_delta_N2_hiiregions', 'std_delta_N2_hiiregions', 'mean_delta_O3_hiiregions', 'std_delta_03_hiiregions', 'mean_delta_ew_hiiregions', 'std_delta_ew_hiiregions', 'mean_delta_N2_dig', 'std_delta_N2_dig', 'mean_delta_O3_dig','std_delta_O3_dig','mean_delta_ew_dig', 'std_delta_ew_dig' )  )


                     
    file_table_info = DIR+'/info_'+name+'.table.ecsv'
    ascii.write(table_info, file_table_info,  overwrite=True, delimiter='&')
    
    print('end')

    return table_info


from astropy.io import fits as pyfits
def RGB_img_cube(fits_file, output_file):
    
    def RGB_img_cube_sim():

        kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]]) / 8.0

        font = { 'size'   : 16}
         
        matplotlib.rc('font', **font)

        ### START rfits_img
    def Slice_Range(wave_1,wave_2):
        global j,nz_med,fitsdata
        global new_cube
        new_cube=1
        wave_now=0.5*(wave_2+wave_1)
        i_med=int((wave_now-crval)/cdelt)
        i1=int((wave_1-crval)/cdelt)
        i2=int((wave_2-crval)/cdelt)
        if (i1>i2):
            i=i2
            i2=i1
            i1=i

        if ((i1>0) and (i2<nz)):
            nz_med=i_med
            tmpdata=fitscube[i1:i2,:,:]
            fitsdata=np.apply_along_axis(sum,0,tmpdata)
            if (i2!=i1):
                fitsdata=fitsdata/(i2-i1)
            fitsdata=np.nan_to_num(fitsdata)
            wave=crval+cdelt*nz_med
            return fitsdata
        

    def rfits_cube(filename):
        global nx,ny,nz,crval,cdelt,crpix
        global Wmin,Wmax,Wmin0,Wmax0
        global new_cube
        new_cube=1
        # READ FITS FILE
        print("Reading cube ",filename)
        fitscube=pyfits.getdata(filename);
        fitshdr=pyfits.getheader(filename);
        nx = fitshdr['NAXIS1']
        ny = fitshdr['NAXIS2']
        nz = fitshdr['NAXIS3']
        try:
            crval = fitshdr['CRVAL3']
        except KeyError:
            crval=1
        try:
            cdelt = fitshdr['CDELT3']
        except KeyError:
            cdelt=1
        crpix = 1.0
        out=np.zeros((nz,ny,nx))    
        infinite=np.isfinite(fitscube,out)
        fitscube=fitscube*out
        fitscube=np.nan_to_num(fitscube)
        Wmin=crval
        Wmax=crval+nz*cdelt
        Wmin0=Wmin
        Wmax0=Wmax
        print("done")
        return fitscube,fitshdr

    def rfits_img(filename):
        # READ FITS FILE
        fitsdata=pyfits.getdata(filename);
        fitshdr=pyfits.getheader(filename);
        nx = fitshdr['NAXIS1']
        ny = fitshdr['NAXIS2']
        out=np.zeros((ny,nx))
        infinite=np.isfinite(fitsdata,out)
        fitsdata=fitsdata*out
        fitsdata=np.nan_to_num(fitsdata)
        return fitsdata,fitshdr

    ### END rfits_img

    def PIL2array(img):
        return np.array(img.getdata(),
                        np.uint8).reshape(img.size[1], img.size[0], 3)

    def array2PIL(arr, size):
        mode = 'RGBA'
        arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
        if len(arr[0]) == 3:
            arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
        return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

    #print sys.argv

    #def plot_img(fig,ax,fitsdata,header):

     #   minval=fitsdata.min
     #   maxval=fitsdata.max
        
      #  logspace = 10.**np.linspace(-1, 4.0, 20)   # 50 equally-spaced-in-log points between 1.e-01 and 1.0e03 
       # cax = ax.imshow(fitsdata, interpolation='nearest', cmap=califa, norm=LogNorm(vmin=0.01, vmax=50)) 
        
       # ax.set_title(header)
       # cbar = fig.colorbar(cax)
       # return

   # out_file=''
   # nargs=len(sys.argv)
   # if (nargs==3):
       # fits_file=sys.argv[1]
       # output_file=sys.argv[2]
   # else:
       # print('USE: RGB_img_cube_flux_elines.py fitsfile output_png')
       # quit()

    #fitsdata,fitshdr=rfits_img(fits_file)
    global j,nx_med,ny_med,nz_med,mask,click1,Obj,listObj
    global filename,fitsdata,fitscube,fitshdr
    global Wmin,Wmax,Wmin0,Wmax0
    global l

    fitscube,fitshdr=rfits_cube(fits_file)
    #fitsdata=Slice_Range(5000,6000)
    Rdata=np.flipud(fitscube[3,:,:]) # NII
    Gdata=np.flipud(fitscube[2,:,:]) # Halpha
    Bdata=np.flipud(fitscube[1,:,:]) # OIII

    #RGBdata_in=np.array((Rdata,Gdata,Bdata))
    #RGBdata=np.swapaxes(RGBdata_in,0,2)
    
    #RGBdata=RGBdata*256.0
    #RGBdata=RGBdata_32.astype('uint8')
    
    print("shape = "+str(Bdata.shape[0])+","+str(Bdata.shape[1]))

    scale_flux=0.25 #2.5
    RGBdata = np.zeros((Bdata.shape[0], Bdata.shape[1], 3), dtype=float)
    #RGBdata[:,:,0] = img_scale.sqrt(Rdata*scale_flux*1.3, scale_min=0.01, scale_max=5)
    #RGBdata[:,:,1] = img_scale.sqrt(Gdata*scale_flux*0.8, scale_min=0.01, scale_max=5)
    #RGBdata[:,:,2] = img_scale.sqrt(Bdata*scale_flux*1.2, scale_min=0.01, scale_max=5)
    RGBdata[:,:,0] = img_scale.sqrt(Rdata*scale_flux*1.2, scale_min=0.01, scale_max=20)
    RGBdata[:,:,1] = img_scale.sqrt(Gdata*scale_flux*0.75, scale_min=0.01, scale_max=20)
    RGBdata[:,:,2] = img_scale.sqrt(Bdata*scale_flux*1.0, scale_min=0.01, scale_max=20)
    RGBshape=RGBdata.shape
    
    #RGBdata = cv2.filter2D(RGBdata, -1, kernel_sharpen_3)
    
    #print 'RGBdata='+str(RGBdata)
    
    #blurred_f = ndimage.gaussian_filter(RGBdata, 3)
    #filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
    #alpha = 30
    #sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    
    #RGBdata=sharpened*256
    #
    RGBdata=RGBdata*256
    #
    # Sharpen
    #
    #
    #
    #
    #RGBdata_int=sharpened.astype('uint8')
    #
    RGBdata_int=RGBdata.astype('uint8')

    img = Image.fromarray(RGBdata_int)
    #img=array2PIL(RGBdata, (Bdata.shape[0], Bdata.shape[1]))
    
    #img=array2image(RGBdata)
    #img=Image.new('RGB',(Bdata.shape[0], Bdata.shape[1]),'white')
    #img.putdata(list([tuple[pixel] for pixel in RGBdata])
    #pixels=img.load()
    #pixels=RGBdata

    print("ShapeRGB ="+str(RGBshape))
    
    contrast = ImageEnhance.Contrast(img)                                                                          
    bright = ImageEnhance.Brightness(img)                                                                          
    img=contrast.enhance(1.75)                                                                                      
    img=bright.enhance(1.75)                
    
    #sharpened.save(output_file,'PNG')
    
    #
    img.save(output_file,'PNG')
    return img.save(output_file,'PNG')
