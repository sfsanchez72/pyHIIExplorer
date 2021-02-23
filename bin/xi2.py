#!/usr/bin/python3
import numpy as np
import scipy as sp
import math
from astropy.io import fits, ascii
from astropy.table import Table, join
import matplotlib.pyplot as plt
from matplotlib import rcParams as rc
from collections import Counter
import re
from matplotlib import rcParams as rc
import matplotlib
import warnings
import argparse

parser = argparse.ArgumentParser(description='###Program to calculate Xi2###', usage='xi2.py name input_file num_pol DIR [--OPTIONAL_ARGUMENTS=*]\nRun with -h for details on the inputs\n ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('name',default='galaxy',type=str, help='Name of the galaxy')
parser.add_argument('input_file', type=str, help='File of max_size')
parser.add_argument('num_pol', default=2, type=int, help='Degree of the polynomial')
parser.add_argument('DIR', default='none', type=str, help='Where save outputfiles')

args = parser.parse_args()

print('Reading files')

name = args.name
input_file = args.input_file
num_pol = args.num_pol
DIR = args.DIR

plt.ion()

#STYLE

rc.update({'font.size': 20,\
           'font.weight': 900,\
           'text.usetex': True,\
           'path.simplify'           :   True,\
           'xtick.labelsize' : 20,\
           'ytick.labelsize' : 20,\
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

#FUNCTION FOR READ FILE

def header_columns(filename,column):
    COMMENT_CHAR = '#'
    col_NAME = []
    with open(filename, 'r') as td:
        for line in td:
            if line[0] == COMMENT_CHAR:
                info = re.split(' +', line.rstrip('\n'))
                col_NAME.append(info[column])
    counts = {k:v for k,v in Counter(col_NAME).items() if v > 1}
    col_NAME_NEW = col_NAME[:]
    for i in reversed(range(len(col_NAME))):
        item = col_NAME[i]
        if item in counts and counts[item]:
            if (counts[item]>1):
                col_NAME_NEW[i] += str(counts[item]-1)
            counts[item]-=1                
    return col_NAME_NEW

col_NAME=header_columns(input_file,2)
xi_2 = ascii.read(input_file, delimiter=',', guess=True, comment="#", 
                     names=col_NAME, fill_values=[('BAD', 'nan')])
#for col in (xi_2.colnames):
#    print(col)

MAX_SIZE_kpc = xi_2['MAX_SIZE_Kpc']
CHI_SQ = xi_2['CHI_SQ']
MAX_SIZE_PARAM = xi_2['MAX_SIZE_PARAM']

p = np.poly1d(np.polyfit(MAX_SIZE_PARAM, CHI_SQ, num_pol))
t = np.linspace(MAX_SIZE_PARAM[0], MAX_SIZE_PARAM[5], 5000)

plt.plot(MAX_SIZE_PARAM, CHI_SQ, 'o', t, p(t), '-')
plt.xlabel('Max size param')
plt.ylabel('Xi$^{2}$')

O = np.around(np.min(p(t)),5)

if num_pol==2:
    l = str(np.min(np.around((p.r).real,3)))
if num_pol!=2:
    l = str(np.around((p.r).real,3))
plt.text(np.min(MAX_SIZE_PARAM),np.max(CHI_SQ)-np.min(CHI_SQ),name+', Xi$^{2}_{min}$: '+str(O)+', '+ 'Max size param: '+l, fontsize=15)

plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9,
                    wspace=0.9, hspace=0.7)
plt.savefig(DIR+'/'+name+'.pdf')

print(l)
print(MAX_SIZE_PARAM[5])
