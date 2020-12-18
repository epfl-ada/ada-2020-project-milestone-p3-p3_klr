import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble, pipeline, model_selection, metrics, preprocessing, svm
from tensorflow import keras
import tensorflow as tf
import seaborn as sns
import pickle

def load_data(data_path='../data/', muchlinski_data=True, fl_data=True, ch_data=True, hs_data=True):
    """
    Function which loads data used by muchlinski, requires
    SambnisImp.csv available here https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KRKWK8&version=1.0
    and notes.txt to be inside specified data_path, specifying paper variables employed
    ----------
    :param data_path: str
        directory in which SambnisImp.csv and notes.txt are placed
    :param muchlinski_data: bool
        whether to load Muchlinski features
    :param fl_data: bool
        whether to load Fearon and Laitin (2003) features
    :param ch_data: bool
        whether to load Collier and Hoeffler (2004) features
    :param hs_data: bool
        whether to load Hegre and Sambanis (2006) features
    :return: dictionary with desired (feature,labels) tuples
    """
    data = {}
    # Extract features used in each paper from notes.txt 
    with open(data_path + 'notes.txt') as f:

        task_notes = f.readlines()
        task_notes = ''.join(task_notes)

    # load data
    master_data = pd.read_csv(data_path + 'SambnisImp.csv', index_col='X')

    # retrieve y
    y_master = master_data['warstds']

    if fl_data:
        fl_feature_names = re.search(r'(?<=Fearon and Laitin \(2003\):)[^.\.]*', task_notes).group()
        fl_feature_names = re.sub('\n|\s|\"', '', fl_feature_names).split(',')
        fl_x = master_data[fl_feature_names]
        data['fl'] = (fl_x, y_master)

    if ch_data:
        ch_feature_names = re.search(r'(?<=Collier and Hoeffler \(2004\):)[^.\.]*', task_notes).group()
        ch_feature_names = re.sub('\n|\s|\"', '', ch_feature_names).split(',')
        ch_x = master_data[ch_feature_names]
        data['ch'] = (ch_x, y_master)

    if hs_data:
        hs_feature_names = re.search(r'(?<=Hegre and Sambanis \(2006\):)[^.\.]*', task_notes).group()
        hs_feature_names = re.sub('\n|\s|\"', '', hs_feature_names).split(',')
        hs_x = master_data[hs_feature_names]
        data['hs'] = (hs_x, y_master)

    if muchlinski_data:
        muchlinski_features = re.search(r'(?<=91 variables:)[^.\(]*', task_notes).group()
        muchlinski_features = re.sub('\n|\s|\"', '', muchlinski_features).split(',')
        muchlinski_x = master_data[muchlinski_features].drop('warstds', axis=1)
        data['muchlinski'] = (muchlinski_x, y_master)

    return data


def perform_gridsearch_cv(X, y, param_grid, pipe, k_folds, scoring, pkl_out, n_jobs, clss_w=None):
    """
    Performs grid search with sklearn's grid_search_cv and sorts results by scoring method
    ----------
    :param X: numpy array or pandas dataframe
    :param y: numpy array or pandas serie
    :param param_grid: dictionary or list of dictionary to perform grid search upon, must be assignable to pipeline steps
    :param pipe: sklearn pipe with steps
    :param k_folds: sklearn cv acceptable parameters, integers or folds object
    :param scoring: str 
        scoring method acceptable by sklearn gridsearchcv
    :param pkl_out: str
        pickle filename including path for output
    :param n_jobs: int
        number of parallel processes to run
    :param clss_w: dict default None
        weights to assign to each class example {0: weight_for_0, 1: weight_for_1} 
    :return: None, saved pickle of results 
    """
    grid_search = model_selection.GridSearchCV(pipe, param_grid, cv=k_folds, scoring=scoring, verbose=1, n_jobs=n_jobs)
    if clss_w is not None:
        grid_search.fit(X, y, clf__class_weight=clss_w)
    else:
        grid_search.fit(X, y)

    gs_res = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),
                        pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=[scoring])], axis=1)

    sorted_gs_res = gs_res.sort_values(by=scoring, ascending=False)
    sorted_gs_res.columns = sorted_gs_res.columns.str.replace('clf__', '')

    sorted_gs_res.to_pickle(pkl_out)

def get_params(method, PICKLE_PATH):
    """
    Loads pickles of results and retrieves hyperparameters
    ----------
    :param method: str
        method which parameters would like to be retrieved for
    :param PICKLE_PATH: str
        directory to path containing pickle files, naming convention imposed gs_rocauc_ + method + "_all.pkl"
    :return: dictionary including hyperparameters
    """
    file = "gs_rocauc_" + method +"_all.pkl"
    with open(PICKLE_PATH + file, "rb") as f:
        params = pickle.load(f)
        params = params.drop('roc_auc',axis=1).iloc[0].to_dict()
    if method == "svm" and np.isnan(params["degree"]):
        params["degree"] = 1
    if "n_estimators" in params:
        params["n_estimators"] = int(params["n_estimators"])
    return params

def save_pkl(object_, file, PICKLE_PATH):
    """
    Saves Object in pickle at target PICKLE_PATH+file
    """
    with open(PICKLE_PATH + file + ".pkl", "wb") as f:
        pickle.dump(object_, f, pickle.HIGHEST_PROTOCOL)
        
def load_pkl(file, PICKLE_PATH, df=True):
    """
    Loads Object from pickle at target PICKLE_PATH+file
    :param df: bool
        whether to return as dataframe or original format
    """
    with open(PICKLE_PATH + file + ".pkl", "rb") as f:
        pkl_obj = pd.DataFrame(pickle.load(f)) if df else pickle.load(f)
    return pkl_obj    

def roc_plt(X, y, pipe, title, k_fold=5, seed=0, create_plot=True):
    """
    Creates RocAuc Curve for given pipe and plots it if desired
    ----------
    :param X: numpy array or pandas dataframe
    :param y: numpy array or pandas serie
    :param pipe: sklearn pipe with steps
    :param title: str
        title to put in plot
    :param k_folds: sklearn cv acceptable parameters, integers or folds object, default 5 folds        
    :param seed: int
        seed for reprodubility, default 0
    :param create_plot: bool
        whether to plot or not
    :return: dictionary containing mean RocAuc values over the folds
    """
    # set seed and ensure input are numpy arrays
    np.random.seed(seed)
    X, y = X.to_numpy(), y.to_numpy()
    cv = model_selection.StratifiedKFold(n_splits=k_fold)

    # initiate empty lists to store fold scoring to aggregate later
    tprs, aucs = [], []

    # x array used to interpolate tprs from the fpr and tpr returned by plot_roc_curve to later determine means over folds
    mean_fpr = np.linspace(0, 1, 100)

    # define subplot and loop through folds retrieving scores on test folds
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (train, test) in enumerate(cv.split(X, y)):
        # fit model
        pipe.fit(X[train], y[train])

        # plot roc-score on each test fold
        viz = metrics.plot_roc_curve(pipe, X[test], y[test],
                                     name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=ax)

        # create scoring backlog of scores, and interpolated tprs for determining means over folds
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr);
        aucs.append(viz.roc_auc)

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
    plt.show() if create_plot else plt.close()

    dict_vars = dict(mean_fpr=mean_fpr, mean_tpr=mean_tpr, mean_auc=mean_auc)

    return dict_vars

def aggregated_roc_plot(df, colors, title, save_path=None, k_folds=5):
    fig, axs = plt.subplots(figsize=(11,8))
    g = sns.lineplot(x='mean_fpr', y='mean_tpr', data=df, hue='Classifier', palette=colors, lw=3)
    g.set_xlabel(f'Mean False Positive Rate over {k_folds} CV test folds')
    g.set_ylabel(f'Mean True Positive Rate over {k_folds} CV test folds')
    g.set_title(title)
    plt.show()
    
    if save_path!=None:
        g.figure.savefig(save_path)
    
METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.AUC(name='auc'),
]


def make_model(nr_features, dropout1, dropout2, optimizer, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(nr_features,)),
        keras.layers.Dropout(dropout1),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(dropout2),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy())

    return model


def NeuralNetwork(build_fn, **kwargs):
    return keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_fn, **kwargs)
