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

data=pd.read_csv('./figure_data/performace_metrics_data.csv')


ax = sns.boxplot(x="Metrics", y="Score", hue="Method", data=data, palette="Set2", linewidth=0.5)

fig = ax.get_figure()

fig.savefig("./saved_figures/box__plot_all_metrics.png", dpi=300)
