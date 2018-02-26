
import logging
import os
import numpy as np

from data_loader import DataLoader




class Scorer(object):


    def second_largest(self,numbers):
        count = 0
        m1 = m2 = float('-inf')
        for x in numbers:
            count += 1
            if x > m2:
                if x >= m1:
                    m1, m2 = x, m1
                else:
                    m2 = x
        return m2 if count >= 2 else None

    def set_df(self,df,column_names, budget, reserve):

        balances = np.ones(len(column_names)) * budget

        df_t = df[column_names]
        df['winner'] = -1
        df['price'] = 0
        logging.info('calculating price paid (second highest). This takes some time...')
        for i in range(len(df)):
            prices = df_t.loc[i].values
            valid1 = balances > prices
            valid2 = prices > 0.0
            valid3 = valid1 * valid2
            new_prices = valid3 * prices
            winner = np.argmax(new_prices)
            price = self.second_largest(new_prices)
            price = np.max([reserve,price])
            df.loc[i, 'price'] = price
            df.loc[i, 'winner'] = winner
            balances[winner] -= price

            if i % 50 == 0:
                logging.info('%f complete',(float(i)/len(df)))

        logging.info('calculating wins by team')
        wins_by_team = df.winner.value_counts()

        teams = df.winner.unique()
        for team in teams:

            t = df.loc[df['winner'] == team]
            clicks = np.sum(t['click'].values)
            logging.info('team ' + str(team) + ' wins:' + str(wins_by_team[team]) + ' clicks:' + str(clicks))

        print(df.head())


if __name__ == '__main__':

    if os.name == 'nt':
        path = '/temp/kaggle/webeconomics/'
    else:
        path = '~/data/web_economics/'

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('starting train_via_rf.py')

    target_col_name = 'click'
    #train_filename = 'train.csv'

    train_filename = 'train.cutdown.csv'

    logging.info('loading DF '+train_filename)
    train_dl = DataLoader()
    train_dl.load_file(path, train_filename)
    df = train_dl.get_df_copy()

    #print(df.head())
    df_t = df[['click']].copy(True)

    df_t['bidTeam1'] = np.random.randint(227, 230, df_t.shape[0])
    df_t['bidTeam2'] = np.ones(df_t.shape[0]) * 278
    df_t['bidTeam3'] = np.random.randint(297, 301, df_t.shape[0])

    # df_t['bidTeam3'] = 3
    # df_t['bidTeam2'] = 2
    # df_t['bidTeam1'] = 1

    s = Scorer()
    s.set_df(df_t,['bidTeam1','bidTeam2','bidTeam3'],300, 227)



