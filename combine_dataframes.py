
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




def add_bidid_to_df():
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


def check_bid_id_on_2_dataframes():
    if os.name == 'nt':
        path = '/temp/kaggle/webeconomics/'
    else:
        path = '~/Downloads/'

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('starting ')

    train_filename1 = 'click_predictions_narrow.csv'
    #train_filename2 = 'elasticnet_basebid.csv'
    train_filename2 = 'rf_pCTR.csv'

    train_dl1 = DataLoader()
    train_dl1.load_file(path, train_filename1)
    df = train_dl1.get_df_copy()
    train_dl1 = None
    gc.collect()


    train_dl2 = DataLoader()
    train_dl2.load_file(path, train_filename2)
    df2 = train_dl2.get_df_copy()
    train_dl2 = None
    gc.collect()

    df2.columns = df2.columns+'_d1'

    for x in df.columns.values:
        df2[x] = df[x]

    print(df2.head())

    x1 = df2['bidid_d1'] == df2['bidid']
    print(sum(x1))


    df3 = df2[['pCTR','click_proba_d1']]
    print(df3.corr())


if __name__ == '__main__':
    check_bid_id_on_2_dataframes()
