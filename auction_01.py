

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import gensim
import pickle
import re
import math
import numpy as np
import logging
import numpy as np
import scipy as sp
import scipy.stats



NUM_BRANDS = 4500
NUM_CATEGORIES = 1250




class MercariProcessor(object):

    __model = None
    __stats_by_category = None

    def __init__(self):

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



    def build_stats_by_category(self, data_path ='../input/mercari-price-suggestion-challenge/'):

        train = pd.read_table(data_path + 'train.csv', sep = ',')
        test = pd.read_table(data_path + 'test.csv', sep=',')
        print('Train shape: ', train.shape)
        print('Test shape: ', test.shape)
        print(train.head(5))

        #    click,weekday,hour,bidid,userid,useragent,IP,region,city,adexchange,domain,url,urlid,slotid,slotwidth,slotheight,slotvisibility,
        # slotformat,slotprice,creative,bidprice,payprice,keypage,advertiser,usertag

        print(train["payprice"].mean())

        print(np.mean(train['payprice'].values))


        for i in range(len(train)):

            if i % 10000 == 0:
                percent = 100.0 * i/float(len(test))
                logging.info(str(percent)+"% complete")

            category_name = str(train.loc[i, 'category_name'])
            if category_name is None:
                category_name = 'NaN'

            price = float(train.loc[i, 'price'])







        logging.info('stats_by_category saved')

    def load_model(self, path):
        logging.info('loading '+path+"dict.pickle")
        pickle_in = open(path+"dict.pickle", "rb")
        self.__stats_by_category = pickle.load(pickle_in)

    def save_model(self,path):
        pickle_out = open(path+"dict.pickle", "wb")
        pickle.dump(self.__stats_by_category, pickle_out)
        pickle_out.close()

def run_unit_tests():
    cat_list = re.split(' |;|/|,|\*|\n', 'sd fkl sdflksdf/sljflas/')
    assert len(cat_list) == 5, "should be 5 elements"
    m = MercariProcessor()


def build_system(data_path):
    m = MercariProcessor()
    m.build_stats_by_category(data_path)

if __name__ == '__main__':
    run_unit_tests()
    build_system('~/data/web_economics/',)

