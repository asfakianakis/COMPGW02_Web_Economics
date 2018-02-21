import logging
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from data_loader import DataLoader


#
#
#
#
def train_rf(df, target_column_name, feature_column_names, train_percentage):
    # based on:
    #  http://dataaspirant.com/2017/06/26/random-forest-classifier-python-scikit-learn/
    #
    # good target values are : click, bidprice, payprice
    #
    # full list of fields is:
    #  click,weekday,hour,bidid,userid,useragent,IP,region,city,adexchange,domain,url,urlid,slotid,slotwidth,slotheight,slotvisibility, slotformat,slotprice,creative,bidprice,payprice,keypage,advertiser,usertag

    print(list(df.columns.values))
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(df[feature_column_names], df[target_column_name],
                                                        test_size=1.0 - train_percentage,
                                                        train_size=train_percentage)
    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)
    print('train ', len(train_x))
    print(list(train_x.columns.values))
    print('test ', len(test_x))
    print(list(test_x.columns.values))
    predictions = clf.predict(test_x)

    print("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print(" Confusion matrix ", confusion_matrix(test_y, predictions))

    logging.info('trained')


if __name__ == '__main__':
    if os.name == 'nt':
        path = '/temp/kaggle/webeconomics/'
    else:
        path = '~/data/web_economics/'

    m = DataLoader()
    # m.load_file(path, 'train.csv')
    # m.load_file(path, 'validation.csv')
    m.load_file(path, 'validation.cutdown.csv')

    df, new_col_names = m.preprocess_datafram(m.get_df_copy())
    feature_column_names = ['weekday', 'hour', 'region', 'city']
    feature_column_names.extend(new_col_names)

    train_rf(df, 'click', feature_column_names, 0.8)
