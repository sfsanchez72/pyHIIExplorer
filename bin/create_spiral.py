#!/usr/bin/python3
import numpy as np
import scipy as sp
from astropy.io import fits
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
from scipy.optimize import minimize, curve_fit, leastsq
import seaborn as sns
import pandas as pd
import warnings
import pandas as pd
from astropy.table import QTable, Table, Column
from astropy import units as u
import matplotlib.image as mpimg
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
def create_bpt_sns_single (x_plt, y_plt, cmap, c, x_min1,x_max1, y_min1, y_max1, legend, namesave, counts):
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
                   shade_lowest=False, alpha=1, marginal_kws={"color": c, "alpha": 0.9}, levels=[0.15, 0.5, 0.99],  joint_kws={'weights':counts})
    
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
                   shade_lowest=False, alpha=0.5, marginal_kws={"color": c1, "alpha": 0.5}, levels=[0.15, 0.5, 0.99])
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

#def create_cont (ny_in, nx_in, h_scale, cont_peak, x_c, y_c, PA, ab):
#    pa = (PA/180)*3.1416
#    img_cont = np.zeros((ny_in,nx_in))
#    for j in np.arange(0, ny_in):
#        for i in np.arange(0, nx_in):
#            x_now = i-x_c
#            y_now = j-y_c
#            x_e = x_now*np.cos(pa)+y_now*np.sin(pa)
#            y_e = x_now*np.sin(pa)-y_now*np.cos(pa)
#            r_dist_now = np.sqrt((y_e*ab)**2+x_e**2)/h_scale
#            img_cont[j,i] = cont_peak*np.exp(-r_dist_now)
#    return img_cont
    
    
def create_cont(ny_in, nx_in, cont, y_sp, x_sp):
  # Crear matriz con ceros del tamaño apropiado
  r = np.zeros((ny_in,nx_in))

  # Rellenar filas multiplo de scale (interpolando entre valores de los elementos de la fila)
  for fil in range(r.shape):
    for col in range(r.shape[1]-1):
      r[fil*y_sp, col*scale:(col+1)*y_sp+1] = np.linspace(cont[fil,col], cont[fil,col+1], x_sp+1)

  # Rellenar resto de ceros, interpolando entre elementos de las columnas
  for fil in range(cont.shape[0]-1):
    for col in range(r.shape[1]):
      r[fil*y_scale:(fil+1)*y_scale + 1, col] = np.linspace(r[fil*y_scale,col], r[(fil+1)*x_scale, col], scale+1)
  return r
    
    
    
    
    
    
    
    
    
    
