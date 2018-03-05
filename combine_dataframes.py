
import logging
import os
import gc
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from data_loader import DataLoader

if __name__ == '__main__':
    if os.name == 'nt':
        path = '/temp/kaggle/webeconomics/'
    else:
        path = '~/data/web_economics/'

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('starting ')

    train_filename1 = 'split_out_from/validation_dummy.csv'
    train_filename2 = 'validation.csv'

    train_dl1 = DataLoader()
    train_dl1.load_file(path, train_filename1)
    df = train_dl1.get_df_copy()
    train_dl1 = None
    gc.collect()


    train_dl2 = DataLoader()
    train_dl2.load_file(path, train_filename2)
    df['bidid'] = train_dl2.get_df_copy()['bidid']
    train_dl2 = None
    gc.collect()
    df.to_csv(path+train_filename1+'.with.bidid.csv', sep=',', index = False)



    #print('average pay price:'+str(np.mean(df['payprice'])))
    #print('average bid price:'+str(np.mean(df['bidprice'])))
