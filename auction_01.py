

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
import matplotlib.pyplot as plt



NUM_BRANDS = 4500
NUM_CATEGORIES = 1250




class MercariProcessor(object):

    __model = None
    __stats_by_category = None

    def __init__(self):

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



    def get_Data(self, data_path):

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
                
        return train, test

    def ExtractColumn(self,data,columnname):
        columns = data.as_matrix(columns=[columnname])
        return columns

    def plot_frequency_xy(x,y):
        graph = plt.figure(figsize=(14,7))
    
        for j in range(2):
            for i in range(2):
                axis = graph.add_subplot(2,2,i*2+j+1)
    
                tit = ''
                if i == 1: tit += 'log'
                else: tit += 'lin'
                tit += ' - '
                if j == 1: tit += 'log'
                else: tit += 'lin'
    
                axis.set_title(tit)
                axis.set_xlabel(var_name)
                axis.set_ylabel('Frequency')
    
                axis.scatter(x, y, color='blue', alpha=0.5)
    
                if i == 1: axis.set_xscale('log')
                if j == 1: axis.set_yscale('log')
    
                plt.grid()
    
        plt.subplots_adjust(hspace=0.35, wspace=0.35)
        plt.show()
        return
    
    def plot_frequency(z):
        z_aux = np.round(z/10)*10
        x_aux, y_aux = np.unique(z_aux, return_counts=True)
        plot_frequency_xy(x_aux, y_aux)
        return
    


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
    train_data, test_data = m.get_Data(data_path)
    variable_data = m.ExtractColumn(train_data,"advertiser")
    print(variable_data)
    #plot_frequency(variable_data)
    

    
    

if __name__ == '__main__':
    run_unit_tests()
    build_system('C:/Users/Akis-/OneDrive/Masters/Web Economics/')

