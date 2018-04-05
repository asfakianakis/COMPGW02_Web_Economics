import logging
from common.data_loader import DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class RFUtil(object):


    def train_model(self, features, labels, n_estimators=100, seed=0, verbose=0, max_depth = 31, max_features=56):
        #model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, oob_score=True, verbose=verbose, max_depth = max_depth, max_features = max_features)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, oob_score=True, verbose=verbose, max_depth = max_depth, max_features = max_features)
        model.fit(features, labels)
        return model

    def tune_rf_param_num_trees(self, train_x, test_x, train_y, test_y):
        # A 95, test accuracy: 55.74%, oob,0.531458576761
        # A 190, test accuracy: 54.54%, oob,0.541214638731
        # 800, test accuracy: 56.56%, oob,0.546098192779
        # 850, test accuracy: 56.56%, oob,0.546200068187

        print('tree count')
        for tree_count in range(900,2000,250):
            model_a = self.train_model(train_x, train_y, n_estimators=tree_count, seed=0, max_depth = 39, max_features = 55)
            accuracy = model_a.score(test_x, test_y)
            print(str(tree_count) + f", test accuracy: {accuracy:0.2%}, oob,"+str(model_a.oob_score_))

    def tune_rf_param_depth(self, train_x, test_x, train_y, test_y):
        #A 31, test accuracy: 55.90 %, oob, 0.532416342784
        #A 39, test accuracy: 54.53%, oob,0.541118602789
        print('Max depth')
        for max_depth in range(30,50,3):
            model_a = self.train_model(train_x, train_y, n_estimators=190, seed=0, max_depth = max_depth, max_features = 55)
            accuracy = model_a.score(test_x, test_y)
            print(str(max_depth) + f", test accuracy: {accuracy:0.2%}, oob,"+str(model_a.oob_score_))

    def tune_rf_param_max_features(self, train_x, test_x, train_y, test_y):
        # A 55, test accuracy: 56.06%, oob,0.53477077174
        print('max_features')
        for max_features in range(5,60,10):
            model_a = self.train_model(train_x, train_y, n_estimators=190, seed=0, max_depth = 31, max_features = max_features)
            accuracy = model_a.score(test_x, test_y)
            print(str(max_features) + f", test accuracy: {accuracy:0.2%}, oob,"+str(model_a.oob_score_))

    def tune_rf(self, a_train, train_percentage, col_names, target_column_name):


        train_x, test_x, train_y, test_y = train_test_split(a_train.loc[:, col_names], a_train[target_column_name],
                                                            test_size=1.0 - train_percentage,
                                                            train_size=train_percentage)

        print('__________________________________')
        self.tune_rf_param_num_trees(train_x, test_x, train_y, test_y)
        print('__________________________________')
        self.tune_rf_param_depth(train_x, test_x, train_y, test_y)
        print('__________________________________')
        self.tune_rf_param_max_features(train_x, test_x, train_y, test_y)
        print('__________________________________')

if __name__ == '__main__':
    if os.name == 'nt':
        path = '/temp/kaggle/webeconomics/'
    else:
        path = '~/data/web_economics/'

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('starting train_via_rf.py')

    target_col_name = 'click'
    train_filename = 'split_out_from/train_dummy.csv.split.0csv'

    train_dl = DataLoader()
    train_dl.load_file(path, train_filename, save_df_as_pickle_file=False)
    df = train_dl.get_df_copy()

    rf_util = RFUtil()
    col_names = []
    col_names.extend(df.columns.values)
    col_names.remove('click')


    rf_util.tune_rf(df,0.8,col_names,'click')
