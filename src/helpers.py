import re 
import pandas as pd

def load_data(data_path='../data/', muchlinski_data=True, fl_data=True, ch_data=True, hs_data=True):
    
    data = {}
    # Extract features used in each paper from notes.txt 
    with open(DATA_PATH+'notes.txt') as f:

        task_notes = f.readlines()
        task_notes = ''.join(task_notes)

    # load data
    master_data = pd.read_csv(DATA_PATH+'SambnisImp.csv', index_col='X')

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