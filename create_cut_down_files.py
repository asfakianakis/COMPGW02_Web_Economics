

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




class FileProcessor(object):

    __model = None
    __stats_by_category = None

    def __init__(self):

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



    def cutdown_files(self, data_path , filename):
        #    click,weekday,hour,bidid,userid,useragent,IP,region,city,adexchange,domain,url,urlid,slotid,slotwidth,slotheight,slotvisibility,
        # slotformat,slotprice,creative,bidprice,payprice,keypage,advertiser,usertag

        df = pd.read_table(data_path + filename, sep=',')
        print('Shape: ', df.shape)

        print(df.head(5))

        df = df.drop('bidid', 1)
        df = df.drop('userid', 1)
        df = df.drop('domain', 1)
        df = df.drop('url', 1)
        df = df.drop('slotid', 1)
        df = df.drop('urlid', 1)
        df = df.drop('keypage', 1)
        df = df.drop('creative', 1)
        df = df.head(50000)
        print(len(df))
        df.to_csv(data_path + filename+'.cutdown.csv', sep=',')

        logging.info('files  saved')

    def process_file(self, data_path, filename):
        #    click,weekday,hour,bidid,userid,useragent,IP,region,city,adexchange,domain,url,urlid,slotid,slotwidth,slotheight,slotvisibility,
        # slotformat,slotprice,creative,bidprice,payprice,keypage,advertiser,usertag

        df = pd.read_table(data_path + filename, sep=',')
        print('Shape: ', df.shape)



        df = df.drop('bidid', 1)
        df = df.drop('userid', 1)
        df = df.drop('domain', 1)
        df = df.drop('url', 1)
        df = df.drop('slotid', 1)
        df = df.drop('urlid', 1)
        df = df.drop('keypage', 1)
        df = df.drop('creative', 1)
        #df = df.head(50000)
        print(len(df))



        usertagset = set()

        for i in range(len(df)):
            if i % 10000 == 0:
                percent = 100.0 * i / float(len(df))
                logging.info(str(percent) + "% complete")
            usertag = str(df.loc[i, 'usertag'])
            usertags = usertag.split(sep=',')
            usertagset.update(usertags)

            #

            #df.loc[i, 'e'] = usertag
            #print(df.head(5))

        print(usertagset)

        logging.info('usertag set: %s' % str(usertagset))

        for usertag in usertagset:
            df[usertag] = pd.Series(np.zeros(len(df)), index=df.index, dtype=int)

        for i in range(len(df)):
            if i % 10000 == 0:
                percent = 100.0 * i / float(len(df))
                logging.info(str(percent) + "% complete")
            usertag = str(df.loc[i, 'usertag'])
            usertags = usertag.split(sep=',')

            for usertag in usertags:
                df.loc[i, usertag] = 1
            #print(df.head(5))

        df.to_csv(data_path + filename + '.usertags_split_out.csv', sep=',')
        logging.info('files  saved')


if __name__ == '__main__':
    m = FileProcessor()
    #m.cutdown_files('~/data/web_economics/', 'test.csv')
    m.process_file('~/data/web_economics/', 'test.csv')

