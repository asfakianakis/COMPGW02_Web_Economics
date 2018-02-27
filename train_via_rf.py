import logging
import os
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from data_loader import DataLoader
#
#
#
#
def train_rf(df, target_column_name, feature_column_names, train_percentage, categorical, n_trees=10, max_features=None, max_depth=None):
    # based on:
    #  http://dataaspirant.com/2017/06/26/random-forest-classifier-python-scikit-learn/
    #
    # good target values are : click, bidprice, payprice
    #
    # full list of fields is:
    #  click,weekday,hour,bidid,userid,useragent,IP,region,city,adexchange,domain,url,urlid,slotid,slotwidth,slotheight,slotvisibility, slotformat,slotprice,creative,bidprice,payprice,keypage,advertiser,usertag

    #print(list(df.columns.values))
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(df[feature_column_names], df[target_column_name],
                                                        test_size=1.0 - train_percentage,
                                                        train_size=train_percentage)
    if (categorical):
        clf = RandomForestClassifier(oob_score=True, n_estimators=n_trees, max_features = max_features, max_depth = max_depth)
    else:
        clf = RandomForestRegressor(oob_score=True, n_estimators=n_trees, max_features = max_features, max_depth = max_depth)

    clf.fit(train_x, train_y)
    logging.info('training size %d ', len(train_x))
    logging.info('testing size %d ', len(test_x))
    predictions = clf.predict(test_x)

    print("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print(" Confusion matrix ", confusion_matrix(test_y, predictions))
    print("oob            :: ", clf.oob_score_)

    logging.info('trained')
    return clf


def predict(model_list, df, feature_column_names, target_column_name = None):
    data_x = df[feature_column_names]
    data_y = None
    if target_column_name in df:
        data_y = df[target_column_name]
    predictions_list = []
    logging.info('Predicting with %d models',len(model_list))
    for model in model_list:
        predictions = model.predict(data_x)
        predictions_list.append(predictions)
    logging.info('Predictions complete')
    net_predictions = np.zeros(len(df))
    if not data_y is None:
        #method 1
        logging.info('method 1 started')
        sum_predictions = np.zeros(len(df))
        for j in range(len(predictions_list)):
            sum_predictions +=  predictions_list[j]
        sum_predictions /= len(predictions_list)
        net_predictions = sum_predictions > 0.5
        logging.info('method 1 complete')
        accuracy = np.sum(net_predictions == data_y) / float(len(data_y))
        print('accuracy %f', accuracy)
        calc_confusion_matrix_binary(data_y, net_predictions)


    df['pCTR'] = net_predictions


    return (df)





def load_model(path, filename):
    logging.info('loading '+path+filename)
    pickle_in = open(path+filename, "rb")
    return(pickle.load(pickle_in))

def save_model(obj, path, filename):
    pickle_out = open(path+filename, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def calc_confusion_matrix_binary(ground_truth, predictions):
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    print('tp %d ', tp)
    print('tn %d ', tn)
    print('fp %d ', fp)
    print('fn %d ', fn)
    return tp, tn, fp, fn


if __name__ == '__main__':

    if os.name == 'nt':
        path = '/temp/kaggle/webeconomics/'
    else:
        path = '~/data/web_economics/'

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('starting train_via_rf.py')

    target_col_name = 'click'
    train_filename = 'split_out_from/train_dummy.csv'
    load_pre_slipt_out_data_sets = True

    expand_train_dataset_columns = False
    validation_filename = 'split_out_from/validation_dummy.csv'
    expand_validation_dataset_columns = False
    max_number_of_datasets = 10 # 1354
    build_models = True  # Took maybe 5 hours to load Achilleas' train and validation sets and train 1300 RFs on them
    save_predictions = False
    save_split_datafroms = False

    model_list = []
    if build_models:
        logging.info('building RF')
        if load_pre_slipt_out_data_sets:
            dataframes = []
            train_dl = DataLoader()
            for i in range(10):
                filename = train_filename + '.split.'+str(i)+'csv'
                train_dl.load_file(path, filename)
                dataframes.append(train_dl.get_df_copy())
        else:
            train_dl = DataLoader()
            train_dl.load_file(path, train_filename, save_df_as_pickle_file=False)
            dataframes = train_dl.get_balanced_datasets(train_dl.get_df_copy(), target_col_name, max_number_of_datasets)
        logging.info('Split into %d dataframes ', len(dataframes))
        if save_split_datafroms:
            i = 0
            for df in dataframes:
                df.to_csv(path + train_filename + '.split.'+str(i)+'csv')
                i += 1

        c = 0
        for df in dataframes:
            if (expand_train_dataset_columns):
                df, new_col_names = train_dl.preprocess_datafram(dataframes[0])
                feature_column_names = ['weekday', 'hour', 'region', 'city']
                feature_column_names.extend(new_col_names)
                #remove columns info could leak from
                feature_column_names = list(filter(lambda a: a != 'click', feature_column_names))
                feature_column_names = list(filter(lambda a: a != 'payprice', feature_column_names))
                feature_column_names = list(filter(lambda a: a != 'bidprice', feature_column_names))
            else:
                feature_column_names = list(df.select_dtypes(include=[np.number]).columns.values) # get only numeric columns
                feature_column_names = list(filter(lambda a: a != 'useragent', feature_column_names))
                feature_column_names = list(filter(lambda a: a != 'usertag', feature_column_names))
                feature_column_names = list(filter(lambda a: a != 'slotid', feature_column_names))
                feature_column_names = list(filter(lambda a: a != 'slotvisibility', feature_column_names))
                #remove columns info could leak from
                feature_column_names = list(filter(lambda a: a != 'click', feature_column_names))
                feature_column_names = list(filter(lambda a: a != 'payprice', feature_column_names))
                feature_column_names = list(filter(lambda a: a != 'bidprice', feature_column_names))
            save_model(feature_column_names, path, 'feature_colum_names_'+str(c)+'.pickle')
            save_model(target_col_name, path, 'target_colum_name_'+str(c)+'.pickle')
            model = train_rf(df, target_col_name, feature_column_names, 0.8, True, max_depth=50, n_trees=900, max_features = 50 )
            save_model(model,path,'rf_model_'+target_col_name+'_'+str(c)+'.pickle')
            model_list.append(model)
            c += 1
            if c > max_number_of_datasets:
                break
    else:
        logging.info('Loading RF')
        for i in range(max_number_of_datasets):
            if i == 0:
                feature_column_names = load_model(path,'feature_colum_names_'+str(i)+'.pickle')
                target_column_name2 = load_model(path,'target_colum_name_'+str(i)+'.pickle')
            fname = 'rf_model_'+target_col_name+'_'+str(i)+'.pickle'
            if os.path.isfile(path+fname):
                model = load_model(path,fname)
                model_list.append(model)

    validation_dl = DataLoader()
    validation_dl.load_file(path, validation_filename)
    val_df = validation_dl.get_df_copy()
    if expand_validation_dataset_columns:
        val_df, new_col_names = validation_dl.preprocess_datafram(val_df)

    missing = set(feature_column_names) - set(val_df.columns.values)
    if (len(missing)>0):
        logging.info('We are missing the following columns:'+str(missing))
        for c in missing:
            val_df[c] = 0.0 # TODO this is a hack?, i.e. setting slotid_PIC to 0? hmm, it would have been anyway
    else:
        logging.info('columns OK')

    df = predict(model_list, val_df, feature_column_names, target_col_name)

    if save_predictions:
        save_model(df, path, target_col_name+'_predictions' + '.pickle')
        df.to_csv(path+target_col_name+'_predictions.csv', sep=',', index = False)
        df2 = df['bidid','pCTR']
        df2.to_csv(path+target_col_name+'_predictions_cutdown.csv', sep=',', index = False)




