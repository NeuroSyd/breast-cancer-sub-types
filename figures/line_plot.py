# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 02:45:36 2019

@author: Raktim Mondol
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#sns.set()
sns.set(style="whitegrid")

data=pd.read_csv('./figure_data/data_precision.csv')



ax = sns.lineplot(x="Classifier", y="Precision", hue="Method",  estimator=None, lw=1.5, palette="Set1", data=data)

fig = ax.get_figure()

fig.savefig("./saved_figures/line_plot_precision.png", dpi=300)

# Rotate the labels on x-axis









