import logging
import os

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


class DataLoader(object):
    __df = None  # data frame of loaded data
    __train_percentage = 0.8

    def __init__(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def split_out_column(self, df, column_name):
        new_columns = []
        user_agent_set = df[column_name].unique()
        logging.info('split into %s', str(user_agent_set))
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
        return (df, new_columns)

    # split out a single collumn into mutiple new columns, that only take the value of 0 or 1.
    # will split fields if needed so 'cat,dog,5' becomes 3 new columns 'x_cat, x_dog, x_5'
    # the '5' above can be ignored (this is the default)
    def split_out_compond_column(self, df, column_name, separator, remove_numeric_values=True):
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
        logging.info('%d tokens will be converted into new columns : %s ', len(token_set), str(token_set))
        logging.info('Estimated time %d seconds == %f minutes ', len(token_set)*8 , len(token_set)*8/60.0)

        # add the new columns
        for v in token_set:
            nf = column_name + '_' + v
            new_columns.append(nf)
            df[nf] = 0
        # populate the new columns
        for v in token_set:
            nf = column_name + '_' + v
            df.loc[df[column_name].str.contains(separator + v + separator), [nf]] = 1
            df.loc[df[column_name].str.startswith(v + separator), [nf]] = 1
            df.loc[df[column_name].str.endswith(separator + v), [nf]] = 1
        return (df, new_columns)

    def load_file(self, data_path, filename):
        #    click,weekday,hour,bidid,userid,useragent,IP,region,city,adexchange,domain,url,urlid,slotid,slotwidth,slotheight,slotvisibility,
        # slotformat,slotprice,creative,bidprice,payprice,keypage,advertiser,usertag
        logging.info('Loading '+data_path+filename)
        self.__df = pd.read_table(data_path + filename, sep=',')
        logging.info('file  loaded')

        # weekday, hour, bidid, userid, useragent, IP, region, city, adexchange, domain, url, urlid, slotid, slotwidth, slotheight, slotvisibility, slotformat, slotprice, creative, bidprice, payprice, keypage, advertiser, usertag

    def save_data_frame(self, df, data_path, filename):
        logging.info('saving df as ' + filename)
        df.to_csv(data_path + filename, sep=',', index=False)

    def preprocess_datafram(self, df):
        #TODO in validation the check with Na fails - so it is not a string - add a type check.
        # replace any Na values
        #if 'slotformat' in df.columns:
        #    df.loc[df['slotformat'] == 'Na', ['slotformat']] = 0
        # break out many value columns into many boolean values columns
        headers = []
        df, new_columns = self.split_out_compond_column(df, 'useragent', separator='_')
        headers.extend(new_columns)
        df, new_columns = self.split_out_compond_column(df, 'usertag', separator=',', remove_numeric_values=False)
        headers.extend(new_columns)
        df, new_columns = self.split_out_compond_column(df, 'slotid', separator='_')
        headers.extend(new_columns)
        df, new_columns = self.split_out_column(df, 'slotvisibility')
        headers.extend(new_columns)
        return df, headers

    def get_df_copy(self):
        return (self.__df.copy(True))


    # return a list of n dataframes, each is balanced, but the less frequent category is fully duplicated into each df returned
    def get_balanced_datasets(self, df, target_column_name):

        compond_set = df[target_column_name].unique()

        if len(compond_set) > 2:
            raise Exception('only supports 2 values at the moment')

        df_a = df[df[target_column_name] == compond_set[0]]
        df_b = df[df[target_column_name] == compond_set[1]]

        if len(df_a) < len(df_b):
            df_lesser = df_a
            df_greater = df_b
        else:
            df_greater = df_a
            df_lesser = df_b


        ratio = int(1+len(df_greater)/len(df_lesser))
        block_size = len(df_lesser)


        dataframes = []
        for i in range(ratio):
            df_t = pd.concat([df_lesser,df_greater[(i*block_size):(i+1)*block_size]])
            dataframes.append(df_t)

        return(dataframes)



if __name__ == '__main__':
    if os.name == 'nt':
        path = '/temp/kaggle/webeconomics/'
    else:
        path = '~/data/web_economics/'

    m = DataLoader()
    # m.load_file(path, 'train.csv')
    m.load_file(path, 'test.csv')
    df, headers = m.preprocess_datafram(m.get_df_copy())
    m.save_data_frame(df, path, 'test.2.csv')


    # dataframes = m.get_balanced_datasets(m.get_df_copy(), 'click')
    # print(len(dataframes))
    # m.train_rf('bidprice')
    # m.train_rf('payprice')
    #m.train_rf('click')
