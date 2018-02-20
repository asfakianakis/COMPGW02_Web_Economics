import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import logging
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class DataLoader(object):

    __df = None  # data frame of loaded data
    __train_percentage = 0.8

    def __init__(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def split_out_column(self, df, column_name):
        new_columns = []
        user_agent_set = df[column_name].unique()
        logging.info('split into %s',str(user_agent_set))
        # add the new columns
        for i in range(len(user_agent_set)):
            v = user_agent_set[i]
            nf = column_name + '_' + v
            new_columns.append(nf)
            df[nf] = 0
        # populate the new columns
        for i in range(len(user_agent_set)):
            v = user_agent_set[i]
            nf = column_name + '_' + v
            df.loc[df[column_name] == v, [nf]] = 1
        return(df,new_columns)

    # split out a single collumn into mutiple new columns, that only take the value of 0 or 1.
    # will split fields if needed so 'cat,dog,5' becomes 3 new columns 'x_cat, x_dog, x_5'
    # the '5' above can be ignored (this is the default)
    def split_out_compond_column(self, df, column_name, separator, remove_numeric_values = True):
        logging.info('Splitting out %s' % column_name)
        new_columns = []

        # gather set of new unique tokens
        compond_set = df[column_name].unique()
        token_set = set()
        for value in compond_set:
            token_set.update(value.split(separator))
        # strip out number i. from a,b,1,2,3,4,5,e only keep a,b,e
        if remove_numeric_values:
            temp_token_set = set()
            for t in token_set:
                if not t.isdigit():
                    temp_token_set.add(t)
            token_set = temp_token_set
        logging.info('%d tokens will be converted into new columns : %s ',  len(token_set), str(token_set))

        # add the new columns
        for v in token_set:
            nf = column_name + '_' + v
            new_columns.append(nf)
            df[nf] = 0
        # populate the new columns
        for v in token_set:
            nf = column_name + '_' + v
            df.loc[df[column_name].str.contains(separator+v+separator), [nf]] = 1
            df.loc[df[column_name].str.startswith(v+separator), [nf]] = 1
            df.loc[df[column_name].str.endswith(separator+v), [nf]] = 1
        return (df, new_columns)


    def load_file(self, data_path, filename):
        #    click,weekday,hour,bidid,userid,useragent,IP,region,city,adexchange,domain,url,urlid,slotid,slotwidth,slotheight,slotvisibility,
        # slotformat,slotprice,creative,bidprice,payprice,keypage,advertiser,usertag
        self.__df = pd.read_table(data_path + filename, sep=',')
        logging.info('file  loaded')

        #weekday, hour, bidid, userid, useragent, IP, region, city, adexchange, domain, url, urlid, slotid, slotwidth, slotheight, slotvisibility, slotformat, slotprice, creative, bidprice, payprice, keypage, advertiser, usertag

    def save_data_frame(self, df, data_path, filename):
        logging.info('saving df as '+filename)
        df.to_csv(data_path + filename, sep=',', index=False)

    def preprocess_datafram(self, df):
        # replace any Na values
        df.loc[df['slotformat'] == 'Na', ['slotformat']] = 0
        # break out many value columns into many boolean values columns
        headers = []
        df, new_columns = self.split_out_compond_column(df,'useragent', separator='_')
        headers.extend(new_columns)
        df, new_columns = self.split_out_compond_column(df,'usertag', separator=',', remove_numeric_values = False)
        headers.extend(new_columns)
        df, new_columns = self.split_out_compond_column(df,'slotid', separator='_')
        headers.extend(new_columns)
        df, new_columns = self.split_out_column(df,'slotvisibility')
        headers.extend(new_columns)
        return df, headers

    def get_df_copy(self):
        return(self.__df.copy(True))


if __name__ == '__main__':
    if os.name == 'nt':
        path = '/temp/kaggle/webeconomics/'
    else:
        path = '~/data/web_economics/'

    m = DataLoader()
    m.load_file(path, 'train.csv')

    #df, headers = m.preprocess_datafram()
    #m.save_data_frame(df, path, 'train.2.csv')

    #m.train_rf('bidprice')
    #m.train_rf('payprice')
    m.train_rf('click')


