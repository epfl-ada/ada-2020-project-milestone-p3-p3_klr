import re 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble, pipeline, model_selection, metrics, preprocessing, svm

def load_data(data_path='../data/', muchlinski_data=True, fl_data=True, ch_data=True, hs_data=True):
    
    data = {}
    # Extract features used in each paper from notes.txt 
    with open(data_path+'notes.txt') as f:

        task_notes = f.readlines()
        task_notes = ''.join(task_notes)

    # load data
    master_data = pd.read_csv(data_path+'SambnisImp.csv', index_col='X')

    # retrieve y
    y_master = master_data['warstds']

    if fl_data:
        fl_feature_names = re.search(r'(?<=Fearon and Laitin \(2003\):)[^.\.]*',task_notes).group()
        fl_feature_names = re.sub('\n|\s|\"','',fl_feature_names).split(',')
        fl_x = master_data[fl_feature_names]
        data['fl'] = (fl_x, y_master)

    if ch_data:
        ch_feature_names = re.search(r'(?<=Collier and Hoeffler \(2004\):)[^.\.]*',task_notes).group()
        ch_feature_names = re.sub('\n|\s|\"','',ch_feature_names).split(',')
        ch_x = master_data[ch_feature_names]
        data['ch'] = (ch_x, y_master)
        
    if hs_data:
        hs_feature_names = re.search(r'(?<=Hegre and Sambanis \(2006\):)[^.\.]*',task_notes).group()
        hs_feature_names = re.sub('\n|\s|\"','',hs_feature_names).split(',')
        hs_x = master_data[hs_feature_names]
        data['hs'] = (hs_x, y_master)
        
    if muchlinski_data:
        muchlinski_features = re.search(r'(?<=91 variables:)[^.\(]*',task_notes).group()
        muchlinski_features = re.sub('\n|\s|\"','',muchlinski_features).split(',')
        muchlinski_x = master_data[muchlinski_features].drop('warstds', axis = 1)
        data['muchlinski'] = (muchlinski_x, y_master)
    
    return data

def roc_plt(X, y, pipe, title, k_fold=5, seed=0):
    
    # set seed and ensure input are numpy arrays
    np.random.seed(seed)
    X, y = X.to_numpy(), y.to_numpy()
    cv = model_selection.StratifiedKFold(n_splits=k_fold)

    # initiate empty lists to store fold scoring to aggregate later
    tprs, aucs = [], []
    
    # x array used to interpolate tprs from the fpr and tpr returned by plot_roc_curve to later determine means over folds
    mean_fpr = np.linspace(0, 1, 100)

    # define subplot and loop through folds retrieving scores on test folds
    fig, ax = plt.subplots(figsize=(12,8))
    for i, (train, test) in enumerate(cv.split(X, y)):
        
        # fit model
        pipe.fit(X[train], y[train])

        # plot roc-score on each test fold
        viz = metrics.plot_roc_curve(pipe, X[test], y[test], 
                                     name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=ax)
        
        # create scoring backlog of scores, and interpolated tprs for determining means over folds
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr) ; aucs.append(viz.roc_auc)

    # red diagonal representing "chance" of ROC-AUC plot - predicting unrelated to the true label
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    # add mean line to plot
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # add standard deviation from mean line
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    # set limits, title & legend and display plot used in this training process
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend(bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.show()
    
    dict_vars = dict(mean_fpr=mean_fpr, mean_tpr=mean_tpr, mean_auc=mean_auc)
    
    return dict_vars