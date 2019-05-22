# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 02:05:04 2018

@author: RaktimMondol
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
dataset = pd.read_csv('./result_data/aae_loss.csv')
x = dataset.iloc[0:100,0].values # no. of epoch
y1 = dataset.iloc[0:100,1].values # training loss single layer
y2 = dataset.iloc[0:100,2].values # validation loss single layer
y3 = dataset.iloc[0:100,3].values # training loss one hidden layer
y4 = dataset.iloc[0:100,4].values # validation loss one hidden layer
y5 = dataset.iloc[0:100,5].values # training loss one hidden layer
y6 = dataset.iloc[0:100,6].values # validation loss two hidden layer

labels = ["Train_Single_Layer", "Validation_Single_Layer", "Train_1_Hidden", "Validation_1_Hidden", "Train_2_Hidden", "Validation_2_Hidden"]
fig, ax = plt.subplots()
ax.plot(x, y1) #plot single layer
ax.plot(x, y2) #plot single layer
ax.plot(x, y3) #plot single layer
ax.plot(x, y4) #plot single layer
ax.plot(x, y5) #plot 1 hidden layer
ax.plot(x, y6) #plot 2 hidden layer
ax.set(xlabel='Training Epoch', ylabel='Loss')
ax.grid()
ax.legend(labels)
fig.savefig('./saved_figures/Training_Loss_plot.png', format='png', dpi=500)
plt.show()