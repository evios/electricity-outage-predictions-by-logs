import pickle
import re
import statistics
import sys

import numpy as np
import pandas as pd
from pandas.api.types import (is_categorical_dtype, is_numeric_dtype,
                              is_string_dtype)
from scipy.special import boxcox1p
from scipy.stats import norm, skew
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# from train import train
# import config as cfg

# sys.path.append('../')

# default='warn', https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html
pd.options.mode.chained_assignment = None


class prod_predict():
    def __init__(self, prod_model_filename):
        # Setup of production arguments and methods
        # All config variables stored in ./config.py
        # self.mp_lv_mv_dict_path = mp_lv_mv_dict_path
        # Init variables
        # Setting new unified names for columns after all manipulations
        self.dev_name_col = 'dev_name'
        self.eval_date_col = 'eval_date'
        self.prod_model_filename = prod_model_filename

    def read_data(self, filenames):
        dfs = ()
        df = pd.DataFrame()

        for file in filenames:
            df = pd.read_feather(file)
            dfs = dfs + tuple((df,))
        df = pd.concat(dfs, axis=0, sort=False, ignore_index=True).fillna(0)

        return df

    def make_hist_df(self, df, period, interval, logs_cols_desc, drop_zeroes=True):
        """
        Suming up events X days (period) before evaluation date
        """
        dev_agg_col = logs_cols_desc[0]
        date_col = logs_cols_desc[1]

        if period >= interval & period % interval == 0:
            
            df_model = pd.DataFrame()
            df_model_interval = pd.DataFrame()

            # get last date in logs # we -1 day for purposes if last day have not full day logs, so we stard the day before last
            last_day_to_look = sorted(df[date_col], reverse=True)[
                0]-pd.DateOffset(days=1)

            # Subset of DF related to period we are interested in
            df = df[df[date_col].between(last_day_to_look-pd.DateOffset(
                days=period), last_day_to_look+pd.DateOffset(days=1), inclusive=False)]

            for count_date in range(1, int(period/interval)+1):
                # will go through day of repeir, -3 days, -3 days, ..., and will count events for last 3 days
                day = last_day_to_look-pd.DateOffset(days=count_date*interval)

                df_model_interval = df[df[date_col].between(day, (day+pd.DateOffset(days=interval)+pd.DateOffset(
                    days=1)), inclusive=False)].drop(columns=date_col).groupby([dev_agg_col]).sum().reset_index()

                # Renaming columns to clarify which period is this column
                df_model_interval = pd.concat([df_model_interval[dev_agg_col], df_model_interval.drop(
                    columns=dev_agg_col).rename(columns=lambda x: (x+': -'+str(count_date*interval)+'day/s'))], sort=False, axis=1)

                if df_model.empty:
                    df_model = df_model_interval
                else:
                    df_model = pd.merge(
                        df_model, df_model_interval, how='outer', on=dev_agg_col).fillna(0)

            df_model = df_model.reset_index(drop=True)
            # For test Input==Output
            original_events_num = round(
                df.drop(columns=[dev_agg_col, date_col]).dropna().values.sum(), 3)
            error = 0
            # Passing test
            cnt_evnts_after_filtering = round(
                df_model.drop(columns=dev_agg_col).values.sum(), 3)
            if original_events_num == cnt_evnts_after_filtering:
                print('Test Succesfully Passed: Data Transormed Correctly, Totally:',
                      original_events_num, ', No events were lost for given DEVs')
                print('Shape of Dataset is: DEVs=',
                      df_model.shape[0], '| Type of events=', df_model.shape[1])
            else:
                error = [original_events_num, cnt_evnts_after_filtering]
                print('Some errors were occured during filtering. Original events number / Events number after filtering',
                      error, 'See error variable for details of how many events were originally and were grouped')

            df_model.rename(
                columns={dev_agg_col: self.dev_name_col}, inplace=True)
            df_model[self.eval_date_col] = last_day_to_look

            # Dropping rows wizh zeroes values
            df_model = self.drop_zeroes_sum(df_model)

            return df_model, error

    def drop_zeroes_sum(self, df):
        df = df[df.drop(columns=[self.dev_name_col,
                                 self.eval_date_col]).sum(axis=1) > 0]

        return df

    def threshold_df_outliers_pre_model(self, df):
        median_df = statistics.median(df.drop(columns=[
                                      self.dev_name_col, self.eval_date_col]).sum(axis=1).sort_values(ascending=False))
        mean_df = statistics.mean(df.drop(columns=[self.dev_name_col, self.eval_date_col]).sum(
            axis=1).sort_values(ascending=False))
        # Throw away LESS than
        df = df[df.drop(columns=[self.dev_name_col, self.eval_date_col]).sum(
            axis=1) >= (median_df/3)]

        # Throw away MORE than
        df = df[df.drop(columns=[self.dev_name_col, self.eval_date_col]).sum(
            axis=1) <= (mean_df*3)]

        df = df.reset_index(drop=True)
        print('Shape of Dataset is: DEVs=',
              df.shape[0], '| Type of events=', df.shape[1])

        return df

    def cyclical_pred(self, df, cycles=5, thresh=0.85, enforce_first_cycle=False, norm_trfm=True, skew_trfm=True):
        for cycle in range(cycles):
            # try: df=df.drop(columns=['Outage_Predicted','Outage Not Expected','Outage Expected'])
            # except: print()

            if cycle == 0 & enforce_first_cycle == True:
                df = self.pred_thresh(df.reset_index(drop=True), 0.97)
            else:
                if cycle != 0:
                    df = df.drop(
                        columns=['Outage_Predicted', 'Outage Not Expected', 'Outage Expected'])
                df = self.pred_thresh(df.reset_index(drop=True), thresh)

            print('Outage predictions left after another :', df.shape[0])

        return df

    def pred_thresh(self, df, filter_coef, norm_trfm=True, skew_trfm=True):
        # load the model from disk
        # filename='../models/finalized_outages_model.sav'
        loaded_model = pickle.load(open(self.prod_model_filename, 'rb'))

        # change '.sav' to '_columns.csv' to open file containing model columns names
        columns_filename = re.sub(
            r'\.sav', '_columns.csv', self.prod_model_filename)

        columns_in_model = list(pd.read_csv(
            columns_filename, header=None).squeeze())

        X = df[columns_in_model]

        if skew_trfm:
            X = self.drop_skew(X)
        if norm_trfm:
            X = self.normalize(X)

        y_pred = loaded_model.predict(X)
        y_pred_proba = loaded_model.predict_proba(X)

        df = pd.concat([df, pd.DataFrame(y_pred, columns=['Outage_Predicted']), pd.DataFrame(
            y_pred_proba, columns=['Outage Not Expected', 'Outage Expected'])], sort=False, axis=1)

        # Filtering / Subsetting
        df = df[df['Outage Expected'] > filter_coef]

        return df

    def normalize(self, df):
        # scaler = preprocessing.RobustScaler()
        scaler = preprocessing.StandardScaler()
        columns = df.columns
        df = pd.DataFrame(scaler.fit_transform(df))
        df.columns = columns

        return df

    def drop_skew(self, df, skew_coef=0.75, lam_coef=0.15):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        skewness = df.select_dtypes(include=numerics).apply(lambda x: skew(x))
        skew_index = skewness[abs(skewness) >= skew_coef].index
        skewness[skew_index].sort_values(ascending=False)
        '''BoxCox Transform'''
        lam = lam_coef

        for column in skew_index:
            df[column] = boxcox1p(df[column], lam)

        return df

    def write_preds(self, df_preds, file):
        df_preds[[self.dev_name_col, self.eval_date_col, 'Outage Expected']].reset_index(
            drop=True).to_excel(('results'+(file.split(sep='\\')[-1])+'.xls'), index=False) # as file variable contains directory as wel we split it an use only filename part
        df_preds.reset_index(drop=True).to_excel(
            'results_detailed.xls', index=False)


    # def evaluate_df_devices_presented_in_real_outages(self, df, otg_date_col, otg_cls_orig):
    #     """
    #     If we have actual outages data for estimating period - we can evaluate model predictions against real historical outages
    #     Usage: evaluate_df_devices_presented_in_real_outages(df_model, lst_lv_outaged)
    #     """
    #     self.otg_date_col=otg_date_col
    #     self.otg_cls_orig=otg_cls_orig
    #     lst_devs_outaged = self.get_outaged_devs_for_period()

    #     if df[df.dev_name.isin(lst_devs_outaged)].shape[0] == 0:
    #         print(
    #             'OOPS, Problem. No Devices from Real Outages presented in list after filtering')
    #     else:
    #         print(round(df[df.dev_name.isin(lst_devs_outaged)].shape[0] /
    #                     len(lst_devs_outaged), 2), ' is in real Outages \n')
    #         print('Missing devices in Logs: \n', set(lst_devs_outaged).difference(
    #             list(df[df.dev_name.isin(lst_devs_outaged)].dev_name)), '\n')
    #         print('Outaged devices Presented in Logs: \n', set(lst_devs_outaged).intersection(
    #             list(df[df.dev_name.isin(lst_devs_outaged)].dev_name)))

    # def get_outaged_devs_for_period(self):
    #     # Reading labels
    #     trn = train.train_model(train_lbls_filename=cfg.train_lbls_filename,
    #                             otg_cols_desc=cfg.otg_cols_desc, logs_cols_desc=cfg.logs_cols_desc)

    #     df_outages = trn.read_labels

    #     # Filter relevant dates
    #     remove_outages_before = df[self.eval_date_col][:1].squeeze()
    #     remove_outages_after = remove_outages_before + pd.DateOffset(days=10)
    #     df_outages = df_outages[(df_outages[otg_date_col] > remove_outages_before) & (
    #         df_outages[otg_date_col] < remove_outages_after)].reset_index(drop=True)

    #     # Filter only Predictable Outages
    #     list_predictable_outages = ['Amortiseerunud', 'Ebaselektiivsus', 'Oht elule/varale',   'Reguleerimata ripped',   'Tulekahju',   'Vale faasijärjestus',
    #                                 'Vale tegevus lülitamisel',  'Ülekoormus', 'Rike tarbija seadmes']
    #     df_outages = df_outages[df_outages[otg_cls_orig].isin(
    #         list_predictable_outages)].reset_index(drop=True)

    #     lst_devs_outaged = list(df_outages['MV_or_LV_name'].unique())

    #     return lst_devs_outaged
