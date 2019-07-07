# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 19:13:32 2018

@author: Raktim Mondol
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('./figure_data/aae_loss.csv', low_memory=False, dtype=float)
x = dataset.iloc[0:100,0].values  #First row then column from dataset

y1= dataset.iloc[0:100,1].values   #First row then column from dataset

y2= dataset.iloc[0:100,2].values   #First row then column from dataset   

y3= dataset.iloc[0:100,3].values   #First row then column from dataset

y4= dataset.iloc[0:100,4].values   #First row then column from dataset 

y5= dataset.iloc[0:100,5].values   #First row then column from dataset

y6= dataset.iloc[0:100,6].values   #First row then column from dataset 



fig, ax = plt.subplots()
plt.ylabel('Loss')
plt.xlabel('Epoch')
ax.errorbar(x, y1, xerr=0.0, yerr=0.0, label='Train_single_layer')
ax.errorbar(x, y2, xerr=0.0, yerr=0.0, label='Validation_single_layer')
ax.errorbar(x, y3, xerr=0.0, yerr=0.0, label='Train_1_hidden')
ax.errorbar(x, y4, xerr=0.0, yerr=0.0, label='Validation_1_hidden')
ax.errorbar(x, y5, xerr=0.0, yerr=0.0, label='Train_2_hidden')
ax.errorbar(x, y6, xerr=0.0, yerr=0.0, label='Validation_2_hidden')
plt.legend(loc='upper right')
plt.savefig('./saved_figures/error_plot.png', format='png', dpi=300)
plt.show()
