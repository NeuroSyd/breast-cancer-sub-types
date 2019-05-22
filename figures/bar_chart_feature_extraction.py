import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets

bar_width=0.15
num_bins=7
PCA=[0.8001,	0.7711,	0.8001,	0.7989,	0.8413,	0.6876,	0.6693]
VAE=[0.7718,	0.7291,	0.7718,	0.7680,	0.8252,	0.6461,	0.6131]
DenoisingAE=[0.7899, 0.7850,	0.7899,	0.7926,	0.8483,	0.6787,	0.6749]
AAE=[0.8545,	0.8538,	0.8545,	0.8574,	0.9017,	0.7761,	0.7748]
AE=[0.8066,	0.7953,	0.8066,	0.7971,	0.8605,	0.7007,	0.6963]
# all result is taken from knn classification


opacity=1
indices=np.arange(num_bins)
fig = plt.figure()
ax = plt.subplot(111)

plt.bar(indices-bar_width, PCA, bar_width, color='b', label='PCA', alpha=opacity)
plt.bar(indices, AAE, bar_width, color='r', label='AAE(Proposed)', alpha=0.9)
plt.bar(indices+bar_width, DenoisingAE, bar_width, color='g', label='DenoisingAE', alpha=opacity)
plt.bar(indices+bar_width+bar_width, VAE, bar_width, color='c', label='VAE', alpha=opacity)
plt.bar(indices+bar_width+bar_width+bar_width, AE, bar_width, color='violet', label='AE', alpha=opacity)

#plt.xlabel('Performace metrics of various dimension reduction techniques are shown')
plt.ylabel('% Score')
plt.xticks(indices+0.145,('Accuracy','F1-Score','Recall','Precision','AUC', 'MCC', 'Kappa'))
plt.tight_layout()
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
          fancybox=True, shadow=False, ncol=5)
plt.savefig('./saved_figures/bar_chart_feature_extraction.png', format='png', dpi=500)
plt.show()