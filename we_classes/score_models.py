
import logging
import os
import numpy as np
import pandas as pd

from we_classes.data_loader import DataLoader

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

        df = df.copy(True)
        balances = np.ones(len(column_names)) * budget
        df_teams_bid_prices_only = df[column_names]
        df['winner'] = -1
        df['price'] = 0
        logging.info('calculating price paid (second highest). This takes some time...')
        for i in range(len(df)):
            prices = df_teams_bid_prices_only.loc[i].values
            valid1 = balances > prices
            valid2 = prices > 0.0
            valid3 = valid1 * valid2
            new_prices = valid3 * prices
            winner = np.argmax(new_prices)
            if (new_prices[winner]>0):
                price = self.second_largest(new_prices)
                price = np.max([reserve,price])
                df.loc[i, 'price'] = price
                df.loc[i, 'winner'] = winner
                balances[winner] -= price

            if i % int(len(df)/49) == 0:
                logging.info('%.2f complete',(float(i)/len(df)))

            if np.sum(balances > 0.0) == 0: # if no balances are positive, we don't need to process the rest of the data frame
                break

        logging.info('calculating wins by team')
        wins_by_team = df.winner.value_counts()
        teams = df.winner.unique()

        team_names = []
        team_wins = []
        team_clicks = []
        team_budgets = []
        for team_index in teams:
            if team_index >= 0:
                temp_df = df.loc[df['winner'] == team_index]
                clicks = np.sum(temp_df['click'].values)
                team_name = df_teams_bid_prices_only.columns[team_index]
                team_names.append(team_name)
                team_wins.append(wins_by_team[team_index])
                team_clicks.append(clicks)
                team_budgets.append(balances[team_index])

                logging.info(
                    'team ' + team_name + ' ' + str(team_index) + ' wins:' + str(wins_by_team[team_index]) + ' clicks:' + str(clicks)+ ' budgets:' + str(balances[team_index]))
        df_summary = pd.DataFrame({"team_name": team_names, "win": team_wins, "click": team_clicks, "budget": team_budgets})
        return(df_summary,df)

'''
if __name__ == '__main__':

    if os.name == 'nt':
        path = '/temp/kaggle/webeconomics/'
    else:
        path = '~/data/web_economics/'

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('starting train_via_rf.py')

    target_col_name = 'click'
    data_filename = 'validation.csv'

    logging.info('loading DF ' + data_filename)
    train_dl = DataLoader()
    train_dl.load_file(path, data_filename)
    df = train_dl.get_df_copy()

    df_t = df[['click']].copy(True)
    df_t['bidTeam1'] = np.random.randint(227, 230, df_t.shape[0])
    df_t['bidTeam2'] = np.ones(df_t.shape[0]) * 278
    df_t['bidTeam3'] = np.random.randint(297, 301, df_t.shape[0])

    print(df_t.head())

    s = Scorer()
    df_summary, df_full = s.set_df(df_t,['bidTeam1','bidTeam2','bidTeam3'],270000000, 227)

    print(df_summary.head())
'''


