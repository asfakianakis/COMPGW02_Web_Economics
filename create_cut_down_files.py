import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import logging
import numpy as np
import os

class FileProcessor(object):
    __model = None
    __stats_by_category = None

    def __init__(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def cutdown_files(self, data_path , filename):
        #    click,weekday,hour,bidid,userid,useragent,IP,region,city,adexchange,domain,url,urlid,slotid,slotwidth,slotheight,slotvisibility,
        # slotformat,slotprice,creative,bidprice,payprice,keypage,advertiser,usertag

        # click, bidprice, payprice

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
        df.to_csv(data_path + filename+'.cutdown.csv', sep=',', index=False)

        logging.info('files  saved')

    def process_file(self, data_path, filename, drop_large_fields = True):
        #    click,weekday,hour,bidid,userid,useragent,IP,region,city,adexchange,domain,url,urlid,slotid,slotwidth,slotheight,slotvisibility,
        # slotformat,slotprice,creative,bidprice,payprice,keypage,advertiser,usertag

        df = pd.read_table(data_path + filename, sep=',')
        print('Shape: ', df.shape)

        #drop the large fields
        if drop_large_fields:
            df = df.drop('bidid', 1)
            df = df.drop('userid', 1)
            df = df.drop('domain', 1)
            df = df.drop('url', 1)
            df = df.drop('slotid', 1)
            df = df.drop('urlid', 1)
            df = df.drop('keypage', 1)
            df = df.drop('creative', 1)

        logging.info('building list of distinct user tags')
        usertagset = set()
        one_percent = int(len(df)/100.0)+1
        one_hundreth_percent = int(one_percent/100.0)+1
        for i in range(len(df)):
            if i % one_percent == 0:
                percent = 100.0 * i / float(len(df))
                logging.info(str(percent) + "% complete")
            usertag = str(df.loc[i, 'usertag'])
            usertags = usertag.split(sep=',')
            usertagset.update(usertags)
        logging.info('usertag set built')

        #add extra fields to dataset
        logging.info('adding %d new columns to dataframe', len(usertagset))
        for usertag in usertagset:
            df[usertag] = pd.Series(np.zeros(len(df)), index=df.index, dtype=int)
        logging.info('new columns added')

        #set extra fields
        logging.info('setting contents of new columns ')
        for i in range(len(df)):
            if i % one_hundreth_percent == 0:
                percent = 100.0 * i / float(len(df))
                logging.info(str(percent) + "% complete")
            usertag = str(df.loc[i, 'usertag'])
            usertags = usertag.split(sep=',')
            for usertag in usertags:
                df.loc[i, usertag] = 1
            #print(df.head(5))
        logging.info('new columns set.')
        # save to file
        df.to_csv(data_path + filename + '.usertags_split_out.csv', sep=',')
        logging.info('files  saved')


if __name__ == '__main__':
    if os.name == 'nt':
        path = '/temp/kaggle/webeconomics/'
    else:
        path = '~/data/web_economics/'


    m = FileProcessor()
    #m.cutdown_files(path, 'test.csv')
    #m.process_file(path, 'test.csv')
    m.process_file(path, 'train.csv', False)

