
############################# IMPORT LIBRARY  #################################
seed=75
import numpy as np
from tensorflow import set_random_seed 
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interp
from itertools import cycle
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import average_precision_score, precision_recall_curve, matthews_corrcoef, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, roc_auc_score, auc, cohen_kappa_score, precision_recall_curve, log_loss, roc_curve, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, cross_val_predict, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics.classification import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn import model_selection
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, MiniBatchSparsePCA, NMF, TruncatedSVD, FastICA, FactorAnalysis, LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import  Normalizer, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, label_binarize, QuantileTransformer
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, RFE, RFECV
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE 
from imblearn.combine import SMOTEENN, SMOTETomek
from keras.initializers import RandomNormal
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from variational_autoencoder import *
from aae_architechture import *
from deep_autoencoder import *
from denoising_autoencoder import *
from shallow_autoencoder import *
matplotlib.use('Agg')
np.random.seed(seed)


#######################   LOAD BREAST CANCER DATASET #######################

file_1 = pd.read_csv('./data/subtype_molecular_rna_seq.csv')
data = file_1.iloc[0:20439,2:607].values  
X=data.T
       
file_2 = pd.read_csv('./data/subtype_molecular_rna_seq_label.csv', low_memory=False)
label= file_2.iloc[0,2:607].values   
y=label.T

print('Actual dataset shape {}'.format(Counter(y)))

############################  LOAD UCEC  DATA   ###########################
'''
file_1 = pd.read_csv('./data/ucec_rna_seq.csv')
data = file_1.iloc[0:20482,2:232].values 
X=data.T

file_2 = pd.read_csv('./data/ucec_rna_seq_label.csv', low_memory=False)
label = file_2.iloc[0,2:232].values   #First row then column from dataset
y=label.T   

print('Actual dataset shape {}'.format(Counter(y)))
'''

count=0
aaecount=0

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 1000)

tprs1 = []
aucs1 = []
mean_fpr1 = np.linspace(0, 1, 1000)

tprs2 = []
aucs2 = []
mean_fpr2 = np.linspace(0, 1, 1000)

tprs3 = []
aucs3 = []
mean_fpr3 = np.linspace(0, 1, 1000)

tprs4 = []
aucs4 = []
mean_fpr4 = np.linspace(0, 1, 1000)
roc_auc_func_macro=[]
roc_auc_func_micro=[]

def zero_mix(x, n):
    temp = np.copy(x)
    noise=n
    if 'spilt' in noise:
        frac = float(noise.split('-')[1])
    for i in temp:
        n = np.random.choice(len(i), int(round(frac * len(i))), replace=False)
        i[n] = 0
    return (temp)

def gaussian_mix(x):
    n = np.random.normal(0, 0.1, (len(x), len(x[0])))
    return (x + n)
# The above two functions are used to add noise in the data
# And used to train denoising autoencoder

skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=seed)
skf.get_n_splits(X, y)
print(skf)
for train_index, test_index in skf.split(X, y):

       #print("TRAIN:", train_index, "TEST:", test_index)
       x_train, x_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
       #print("TRAIN:", x_train, "TEST:", x_test)
       print('Dataset shape for Train {}'.format(Counter(y_train)))
       print('Dataset shape for Test {}'.format(Counter(y_test)))
       
           
       ################################# OVER SAMPLING ###############################
      
       sm = SMOTE(sampling_strategy='auto', kind='borderline1', random_state=seed)
       x_train, y_train = sm.fit_sample(x_train, y_train)
       #oversample only traning data
      
       print('Resampled dataset shape for Train {}'.format(Counter(y_train)))
       print('Resampled dataset shape for Test {}'.format(Counter(y_test)))
       
       
       #use this when train denoising autoencoder
       #use either gaussian mix or zero mix

       #x_train_noisy=zero_mix(x_train, 'spilt-0.05')
       #x_test_noisy=zero_mix(x_test, 'spilt-0.05')
      
       #x_train_noisy=gaussian_mix(x_train)
       #x_test_noisy=gaussian_mix(x_test)
       #n_samples, n_features = x_train.shape

       #############################  FEATURE SCALING/NORMALIZATION ##################

       qt = QuantileTransformer(n_quantiles=10, random_state=seed)
       qt.fit(x_train)
       x_train=qt.transform(x_train)
       x_test=qt.transform(x_test)

      
       '''
       # Standart Scaling
       sc = StandardScaler()
       sc.fit(x_train)
       x_train=sc.transform(x_train)
       x_test=sc.transform(x_test)
       '''   
          
       ###############################DIMENSION REDUCTION ############################
      
       '''
       pca = PCA(n_components=50, random_state=seed)
       pca.fit(x_train, y_train)
       x_train = pca.transform(x_train)
       x_test = pca.transform(x_test)
       print ('After PCA', x_train.shape)
       '''
       
       
       ################ VARIOUS AUTOENCODERS ###############
       
       aaecount= aaecount+1
       aaenum=str(aaecount)


       ######### Shallow Autoencoder ############
       '''
       shallow_autoencoder_fit(x_train, x_test, encoding_dim=50, optimizer="adadelta",
       						   loss_function="binary_crossentropy", nb_epoch=100, 
       						   batch_size=20, path='./feature_extraction/shallowAE/'+aaenum+'/')

       ####### don't use the following lines when autoencoder requires fine tuning
       shallow_autoencoder = load_model('./feature_extraction/shallowAE/'+aaenum+'/shallow_encoder'+'.h5')
       x_train = shallow_autoencoder.predict(x_train)
       print('X_Train Shape after ShallowAE :', x_train.shape)
       x_test = shallow_autoencoder.predict(x_test)
       print('X_Test Shape after ShallowAE :', x_train.shape)
       '''
       

       ######### Denoising Autoencoder ############
       '''
       denoising_autoencoder_fit(x_train, x_test, x_train_noisy, x_test_noisy, encoding_dim=50, optimizer="adadelta",
       							 loss_function="binary_crossentropy", nb_epoch=50, 
       							 batch_size=20, path='./feature_extraction/denoisingAE/'+aaenum+'/')
       #after fiting the model once; it can be directly load from the saved folder hence you can comment out the above lines
       #do not require fine tuning since this autoencoder does not have any hidden layer
       denoising_autoencoder = load_model('./feature_extraction/denoisingAE/'+aaenum+'/denoising_encoder'+'.h5')

       x_train = denoising_autoencoder.predict(x_train)
       print('X_Train Shape after ShallowAE :', x_train.shape)
       x_test = denoising_autoencoder.predict(x_test)
       print('X_Test Shape after ShallowAE :', x_train.shape)
       '''
       

       ######### Deep Autoencoder  ##########
       '''
       deep_autoencoder_fit(x_train, x_test, encoding_dim=50, optimizer="adadelta",
                            loss_function="binary_crossentropy", nb_epoch=100, 
                            batch_size=20, path='./feature_extraction/DeepAE/'+aaenum+'/')
       
       ####### don't use the following lines when autoencoder requires fine tuning
       deep_encoder = load_model('./feature_extraction/DeepAE/'+aaenum+'/deep_autoencoder'+'.h5')
       
       x_train = deep_encoder.predict(x_train)
       print('X_Train Shape after DeepAE :', x_train.shape)
       
       x_test = deep_encoder.predict(x_test)
       print('X_Test Shape after DeepAE :', x_test.shape)
       '''
       
       ##############  AAE  ##############
       
       aae_model('./feature_extraction/AAE/'+aaenum+'/', AdversarialOptimizerSimultaneous(),
                 xtrain=x_train,ytrain=y_train, xtest=x_test, ytest=y_test, encoded_dim=50,img_dim=x_train.shape[1], nb_epoch=100)          
       
       '''
       ####### don't use the following lines when autoencoder requires fine tuning
       model = load_model('./feature_extraction/AAE/'+aaenum+'/aae_encoder'+'.h5')

       x_train = model.predict(x_train)
       print('X_Train Shape after AAE :', x_train.shape)
       
       x_test = model.predict(x_test)
       print('X_Test Shape after AAE :', x_test.shape)
       '''
                 
       ################  Variational Autoencoder  ####################
       '''
       vae_model_single('./feature_extraction/VAE/'+aaenum+'/',x_train.shape[1],
       					x_train,x_test,intermediate_dim=1000,batch_size=20,latent_dim=50,epochs=50)
       '''
       '''
       ####### don't use the following lines when autoencoder requires fine tuning
       model = load_model('./feature_extraction/VAE/'+aaenum+'/vae_encoder'+'.h5')
       x_train = model.predict(x_train)
       print('X_Train Shape after VAE :', x_train.shape)
       x_test = model.predict(x_test)
       print('X_Test Shape after VAE :', x_test.shape)
       '''
       
       ########################    CLASSIFICATION    ##########################
       
       ####### use one classifier at a time  ########
       clf=RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=seed)
       
             
       ###################################################################
       ## Multiclass ROC with Cross Val
       ###################################################################
       
       y_test = label_binarize(y_test, classes=[0,1,2,3])
       
       clf.fit(x_train,y_train)
       y_score=clf.predict_proba(x_test)
       n_classes = y_score.shape[1]
       
       print('ROC_AUC', roc_auc_score(y_test,y_score, average='macro'))
       roc_auc_func_macro.append(roc_auc_score(y_test,y_score, average='macro'))
       roc_auc_func_micro.append(roc_auc_score(y_test,y_score, average='micro'))
       # Compute ROC curve and ROC area for each class
       fpr = dict()
       tpr = dict()
       roc_auc = dict()
       
       # Compute micro-average ROC curve and ROC area
       
       fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
       roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
       
       tprs.append(interp(mean_fpr, fpr["micro"], tpr["micro"]))
       tprs[-1][0] = 0.0
       aucs.append(roc_auc["micro"])
       
       
       fpr["class1"], tpr["class1"], _ = roc_curve(y_test[:, 0], y_score[:, 0])
       roc_auc["class1"] = auc(fpr["class1"], tpr["class1"])
       tprs1.append(interp(mean_fpr1, fpr["class1"], tpr["class1"]))
       tprs1[-1][0] = 0.0
       aucs1.append(roc_auc["class1"])
       
       
       fpr["class2"], tpr["class2"], _ = roc_curve(y_test[:, 1], y_score[:, 1])
       roc_auc["class2"] = auc(fpr["class2"], tpr["class2"])
       tprs2.append(interp(mean_fpr2, fpr["class2"], tpr["class2"]))
       tprs2[-1][0] = 0.0
       aucs2.append(roc_auc["class2"])
       
       
       
       fpr["class3"], tpr["class3"], _ = roc_curve(y_test[:, 2], y_score[:, 2])
       roc_auc["class3"] = auc(fpr["class3"], tpr["class3"])       
       tprs3.append(interp(mean_fpr3, fpr["class3"], tpr["class3"]))
       tprs3[-1][0] = 0.0
       aucs3.append(roc_auc["class3"])
       
       
       fpr["class4"], tpr["class4"], _ = roc_curve(y_test[:, 3], y_score[:, 3])
       roc_auc["class4"] = auc(fpr["class4"], tpr["class4"])
       tprs4.append(interp(mean_fpr4, fpr["class4"], tpr["class4"]))
       tprs4[-1][0] = 0.0
       aucs4.append(roc_auc["class4"])
       
       
             
################################################################################


plt.figure()
lw = 2
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)
       
mean_tpr1 = np.mean(tprs1, axis=0)
mean_tpr1[-1] = 1.0
mean_auc1 = auc(mean_fpr1, mean_tpr1)
std_auc1 = np.std(aucs1)
plt.plot(mean_fpr1, mean_tpr1, color='aqua',
         label='ROC curve for LumA (area = %0.2f $\pm$ %0.2f)' % (mean_auc1, std_auc1),
         lw=2, alpha=.8)


mean_tpr2 = np.mean(tprs2, axis=0)
mean_tpr2[-1] = 1.0
mean_auc2 = auc(mean_fpr2, mean_tpr2)
std_auc2 = np.std(aucs2)

plt.plot(mean_fpr2, mean_tpr2, color='brown',
         label='ROC curve for LumB(area = %0.2f $\pm$ %0.2f)' % (mean_auc2, std_auc2),
         lw=2, alpha=.6)



mean_tpr3 = np.mean(tprs3, axis=0)
mean_tpr3[-1] = 1.0
mean_auc3 = auc(mean_fpr3, mean_tpr3)
std_auc3 = np.std(aucs3)

plt.plot(mean_fpr3, mean_tpr3, color='olive',
         label='ROC curve for Basal & Tri-Neg (area = %0.2f $\pm$ %0.2f)' % (mean_auc3, std_auc3),
         lw=2, alpha=.8)


mean_tpr4 = np.mean(tprs4, axis=0)
mean_tpr4[-1] = 1.0
mean_auc4 = auc(mean_fpr4, mean_tpr4)
std_auc4 = np.std(aucs4)


plt.plot(mean_fpr4, mean_tpr4, color='orange',
         label='ROC curve for Her2 (area = %0.2f $\pm$ %0.2f)' % (mean_auc4, std_auc4),
         lw=2, alpha=.8)


############## Micro Mean   ################
'''
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='black',
         label=r'Micro Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=3, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.1,
                 label=r'$\pm$ 1 std. dev.')

'''
###################### Macro Mean  #####################

all_fpr = np.unique(np.concatenate([mean_fpr1, mean_fpr2, mean_fpr3, mean_fpr4]))

# Then interpolate all ROC curves at this points
mean_tpr_macro = np.zeros_like(all_fpr)
mean_tpr_macro[-1] = 1.0
mean_tpr_macro = np.mean(tprs1, axis=0)+np.mean(tprs2, axis=0)+np.mean(tprs3, axis=0)+np.mean(tprs4, axis=0)


# Finally average it and compute AUC
mean_tpr_macro /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr_macro
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
         ''.format(roc_auc["macro"]),
         color='black', linestyle='-', linewidth=3, alpha=0.8)

##################### Plot Graph ########################
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
plt.savefig('./figures/ROC_Curve.png', format='png', dpi=200)
plt.show()
