import os
from data_loader import DataLoader
import tensorflow as tf
import time
import sys
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().magic('matplotlib inline')

from sklearn.datasets import load_boston


# ref
#  https://stackoverflow.com/questions/45262280/multiple-regression-on-tensorflow



#
#
# Load data
#
#

#
# n_samples = train_X.shape[0]





def train_tf(df, target_column_name, feature_column_names, param1):
    t1 = time.time()
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)  # tf.logging.info('Flags: %s ', FLAGS)

    rng = numpy.random

    # Parameters
    learning_rate = 1e-5
    training_epochs = 10000

    X_data = df[feature_column_names].as_matrix()
    y_data = df[target_column_name].as_matrix()

    m = len(X_data)
    n = len(X_data[0])

    X = tf.placeholder(tf.float32, [m, n])
    y = tf.placeholder(tf.float32, [m, 1])

    W = tf.Variable(tf.ones([n, 1]))
    b = tf.Variable(tf.ones([1]))

    y_ = tf.matmul(X, W) + b

    loss = tf.reduce_mean(tf.square(y - y_))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        vals = []
        for i in range(training_epochs):
            val = sess.run([train, loss], feed_dict={X: X_data, y: y_data[:, None]})
            vals.append(val)
            print(val)

    print(vals)


if __name__ == '__main__':
    if os.name == 'nt':
        path = '/temp/kaggle/webeconomics/'
    else:
        path = '~/data/web_economics/'

    m = DataLoader()
    #m.load_file(path, 'train.csv')
    #m.load_file(path, 'validation.csv')
    m.load_file(path, 'validation.cutdown.csv')

    df, new_col_names  = m.preprocess_datafram(m.get_df_copy())
    feature_column_names = ['weekday', 'hour', 'region', 'city']
    feature_column_names.extend(new_col_names)

    train_tf(df, 'click', feature_column_names, 0.8)


