import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets

bar_width=0.15
num_bins=7
LDA=[0.8983,	0.8989,	0.8983,	0.9150,	0.9681,	0.8667,	0.8609]
LR=[0.8918,	0.8902,	0.8918,	0.9045,	0.9815,	0.8575,	0.8520]
RF=[0.8827,	0.8811,	0.8827,	0.9011,	0.9708,	0.8470,	0.8397]
XGBOOST=[0.8655, 0.8635,	0.8655, 0.8778,	0.9718,	0.8211,	0.8150]
MLP=[0.8866,	0.8848,	0.8866,	0.9002, 0.9794,	0.8508, 0.8449]

opacity=1
indices=np.arange(num_bins)
fig = plt.figure()
ax = plt.subplot(111)

plt.bar(indices-bar_width, LDA, bar_width, color='blue', label='LDA', alpha=0.8)
plt.bar(indices, LR, bar_width, color='gold', label='LR', alpha=opacity)
plt.bar(indices+bar_width, RF, bar_width, color='gray', label='RF', alpha=opacity)
plt.bar(indices+bar_width+bar_width, XGBOOST, bar_width, color='green', label='XGBOOST', alpha=opacity)
plt.bar(indices+bar_width+bar_width+bar_width, MLP, bar_width, color='brown', label='MLP', alpha=opacity)

#plt.xlabel('Performace metrics of various dimension reduction techniques are shown')
plt.ylabel('% Score')
plt.xticks(indices+0.145,('Accuracy','F1-Score','Recall','Precision','AUC', 'MCC', 'Kappa'))
plt.tight_layout()
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
          fancybox=True, shadow=False, ncol=5)
plt.savefig('./saved_figures/bar_chart_classifier.png', format='png', dpi=500)
plt.show()