# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 18:01:47 2017

@author: wangronin
"""

import os, pdb
import matplotlib.pyplot as plt
from matplotlib import rcParams

from scipy.interpolate import  interp1d
import numpy as np

import pandas as pd


rcParams['legend.numpoints'] = 1
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['xtick.major.size'] = 10
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.size'] = 10
rcParams['ytick.major.width'] = 1
rcParams['axes.labelsize'] = 30
rcParams['font.size'] = 30
rcParams['lines.markersize'] = 11
rcParams['xtick.direction'] = 'out'	
rcParams['ytick.direction'] = 'out'

os.chdir(os.path.expanduser('~') + '/Desktop/EGO_python/data')

plt.style.use('ggplot')
fig_width = 22
fig_height = fig_width * 9 / 16

# data_files = ['2D-500N-griewank-BFGS.csv', '2D-500N-griewank-BFGS-tree.csv', '2D-500N-griewank-CMA-tree.csv',
#               '2D-500N-griewank-CMA.csv']

data_files = ['EI_2D_100run.csv', 'MGF_2D_100run.csv']

color = ['k', 'b', 'r', 'c']
marker = ['^', 'o', 's', '*']

fig0, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), subplot_kw={'aspect':'auto'}, dpi=100)

line = []
for i, f in enumerate(data_files):
    df = pd.read_csv(f)  
    
    y, sd = df.loc[:, 'run1':].mean(1), df.loc[:, 'run1':].std(1) / 4.
    
    x = np.arange(1, len(y)+1)
    line += ax.plot(x, y, ls='-', lw=2, color=color[i], marker=marker[i], ms=7, 
                mfc='none', mec=color[i], mew=1.5, alpha=0.6)
    
    f1 = interp1d(x, y-sd, kind='cubic')
    f2 = interp1d(x, y+sd, kind='cubic')
    
    ax.fill_between(x, f1(x), f2(x), facecolor=color[i], alpha=0.2, interpolate=True)

ax.legend(line, map(lambda f: f.split('.')[0], data_files), fontsize=10)
ax.set_yscale('log')
                                
plt.show()
