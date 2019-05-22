# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 16:24:29 2018

@author: Raktim Mondol
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")

data=pd.read_csv('./result_data/log_loss_data.csv')

ax = sns.boxplot(x="Metrics", y="Score", hue="Method", data=data, palette="Set2", linewidth=2)

fig = ax.get_figure()
fig.savefig("./saved_figures/box_plot_log_loss.png", dpi=300)