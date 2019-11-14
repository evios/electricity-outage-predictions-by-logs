import pickle
import re
import statistics

import numpy as np
import pandas as pd
from pandas.api.types import (is_categorical_dtype, is_numeric_dtype,
                              is_string_dtype)
from scipy.special import boxcox1p
from scipy.stats import norm, skew
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

# default='warn', https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html
pd.options.mode.chained_assignment = None


class train_model():
    def __init__(self, train_lbls_filename, otg_cols_desc, logs_cols_desc):
        # Setup of production arguments and methods
        # All config variables stored in ./config.py
        # self.mp_lv_mv_dict_path = mp_lv_mv_dict_path
        # Init variables
        # Setting new unified names for columns after all manipulations
        self.train_lbls_filename = train_lbls_filename

        # Columns names releted to outages
        self.otg_date_col = otg_cols_desc[0]
        self.otg_cls_orig = otg_cols_desc[1]
        self.otg_cls_model = otg_cols_desc[2]
        self.otg_trn_good = otg_cols_desc[3]
        self.otg_mv_lv_name_col = otg_cols_desc[4]
        self.otg_affect_cust_col = otg_cols_desc[5]

        # Columns names releted to logs
        self.dev_agg_col = logs_cols_desc[0]
        self.date_col = logs_cols_desc[1]
        self.event_col = logs_cols_desc[2]
        self.actual_log_collector_dev_col = logs_cols_desc[3]

        # Columns names releted to output DF for model
        self.dev_name_col = 'dev_name'
        self.eval_date_col = 'eval_date'

    """
    Mode = train
    """

    def read_data(self, filenames, df_lbl_outages):

        dfs = ()
        df = pd.DataFrame()

        lst_lv_outages = df_lbl_outages.lv_substation_name.dropna().unique()

        for file in filenames:
            df = pd.read_feather(file)

            # Filter only outages on LV Level
            # df[self.dev_agg_col]=df[self.dev_agg_col].str.replace(' ', '') # Removing spaces from names as we did in dictionary
            # df= rm_spaces(df,[self.dev_agg_col])
            # getting subset for Train LV
            df = df[df[self.dev_agg_col].isin(lst_lv_outages)]

            dfs = dfs + tuple((df,))
        df = pd.concat(dfs, axis=0, sort=False, ignore_index=True).fillna(0)

        df[self.date_col] = pd.to_datetime(df[self.date_col])

        return df

    def get_train_lbl(self, get_unq_devs=False):
        df_lbl_outages = pd.read_csv(
            self.train_lbls_filename, parse_dates=[self.otg_date_col])

        if get_unq_devs != True:
            # Return full labeled DataFrame
            return df_lbl_outages

        else:
            # Return only unique device names to sub filter DF
            lst_lv_outages = df_lbl_outages[self.dev_agg_col].dropna().unique()
            return lst_lv_outages

    def make_hist_df_train(self, df, period, interval):
        df_model = pd.DataFrame()

        # Get training labels
        df_labels = self.get_train_lbl()

        for index, row in df_labels.iterrows():
            df_model_interval = pd.DataFrame()

            # last_day_to_look=row[self.otg_date_col]
            # Trying to remove data leakage of Power Down event connected to Outages
            # removing last 1 days to remove leakege possibly may be containing in logs
            last_day_to_look = row[self.otg_date_col]-pd.DateOffset(days=1)

            print(index, row[self.otg_date_col],
                  last_day_to_look, row[self.dev_agg_col])

            # will go through day of repeir, -3 days, -3 days, ..., and will count events for last 3 days
            day = last_day_to_look-pd.DateOffset(days=interval)

            df_model_interval = df[(df[self.dev_agg_col] == row[self.dev_agg_col]) & (df[self.date_col].between(day, (day+pd.DateOffset(
                days=interval)+pd.DateOffset(days=1)), inclusive=False))].drop(columns=self.date_col).groupby([self.dev_agg_col]).sum().reset_index()

            print(day, day+pd.DateOffset(days=interval)+pd.DateOffset(days=1))

            # Renaming columns to clarify which period is this column
            df_model_interval = df_model_interval.drop(columns=self.dev_agg_col).rename(
                columns=lambda x: (x+': -'+str(interval)+'day/s'))

            df_model_interval[self.dev_agg_col] = row[self.dev_agg_col]

            # Test
            cnt_evnts_after_filtering = round(
                df_model_interval.drop(columns=self.dev_agg_col).values.sum(), 3)
            print('Totally:', cnt_evnts_after_filtering, '')

            df_model_interval[self.eval_date_col] = last_day_to_look
            df_model_interval['y_label_class'] = row[self.otg_trn_good]

            # Adding usuful train data
            df_model_interval[self.otg_cls_orig] = row[self.otg_cls_orig]
            df_model_interval[self.otg_cls_model] = row[self.otg_cls_model]

            df_model = pd.concat(
                [df_model, df_model_interval], sort=False, ignore_index=True)

        df_model.rename(
            columns={self.dev_agg_col: self.dev_name_col}, inplace=True)

        return df_model

    def prep_train_df_model(self, df, drp_low_var_ft=True, norm_trfm=False, skew_trfm=False):
        leEIC = LabelEncoder()
        leEIC.fit(df[self.dev_name_col])
        df[self.dev_name_col] = leEIC.transform(df[self.dev_name_col])
        X = df.copy()

        # Making X and y train DFs
        X.drop(columns=[self.dev_name_col, self.eval_date_col, 'y_label_class',
                        self.otg_cls_orig, self.otg_cls_model], inplace=True)
        y = df.y_label_class

        if drp_low_var_ft:
            X = self.drop_low_var(X)
        if skew_trfm:
            X = self.drop_skew(X)
        if norm_trfm:
            X = self.normalize(X)

        return X, y

    def drop_low_var(self, df):
        # Saving all features for future comparison.
        all_features = df.keys()
        # Removing features.
        df = df.drop(df.loc[:, (df == 0).sum() >= (df.shape[0]*0.995)], axis=1)
        df = df.drop(df.loc[:, (df == 1).sum() >= (df.shape[0]*0.995)], axis=1)
        # Getting and printing the remaining features.
        remain_features = df.keys()
        remov_features = [
            st for st in all_features if st not in remain_features]
        print(len(remov_features), 'features were removed')

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

    def train_save_modeling(self, X, y, prod_model_filename):
        # Saving columns names
        # change '.sav' to '_columns.csv' to open file containing model columns names
        columns_filename = re.sub(
            r'\.sav', '_columns.csv', prod_model_filename)
        pd.Series(list(X.columns)).to_csv(
            columns_filename, index=False, header=False)

        # Train on full DF
        model = XGBClassifier(
            learning_rate=0.1, n_estimators=400, max_depth=20)
        model.fit(X, y)

        # save the model to disk
        filename = 'models/finalized_outages_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        print('Model succesfully created and saved to disk')

    """
    Mode = prep_dict_lbls
    """

    def build_dict(self, filenames, devs_cols, filename_dict_to_wr):
        df = pd.DataFrame()

        for file in filenames:
            # df = pd.read_feather(file)
            df = pd.read_csv(file,
                             parse_dates=[self.date_col])

            df = df[devs_cols]
            gk = df.groupby(self.actual_log_collector_dev_col)
            dic_tmp = gk.first().reset_index()
            try:
                df_dict = pd.merge(df_dict, dic_tmp, on=devs_cols, how='outer')
            except:
                df_dict = dic_tmp

        # Counting number of logging devices on each upper level device
        for dev in devs_cols[1:]:
            df_grouped = df_dict.drop(columns=df_dict.columns.difference([dev, self.actual_log_collector_dev_col])).pivot_table(
                index=[dev], aggfunc=len, fill_value=0).reset_index().rename(columns={self.actual_log_collector_dev_col: 'eics_on_'+dev[:2]})
            df_dict = pd.merge(df_dict, df_grouped, on=dev)

        df_dict.to_feather(filename_dict_to_wr)
        # writing just for Readability purpose and Other projects. In current will not use CSV dictionary
        df_dict.to_csv((filename_dict_to_wr+'.csv'), index=False)
        return df_dict

    def get_outages(self):

        for file in self.outages_files:
            df = pd.read_excel(file[0], usecols=file[1:])
            new_col_names = [self.otg_date_col, self.otg_affect_cust_col,
                self.otg_mv_lv_name_col, self.otg_cls_orig]
            cols_to_rename = dict(zip(file[1:], new_col_names))
            try: df_outages = df_outages.append(df.rename(columns=cols_to_rename), sort=False)
            except: df_outages = df.rename(columns=cols_to_rename)

        df_outages[self.otg_date_col] = pd.to_datetime(
            df_outages[self.otg_date_col], utc=True)
        df_outages[self.otg_date_col] = pd.DatetimeIndex(
            df_outages[self.otg_date_col]).tz_convert(None)

        print('Shape of readed outages DF: ', df_outages.shape)    
        return df_outages


    def prep_labels(self, full_struct_dict_path, list_predictable_outages, list_random_outages, list_unknown_outages, list_synth_good, outages_files, filter_otg_dates, devs_cols):
        self.outages_files = outages_files
        df_outages = self.get_outages()

        dev_level_up_1 = devs_cols[1]   # LV - low voltage substation
        dev_level_up_2 = devs_cols[2]   # MV - middle voltage substation

        """
        Filter outages by period of logs we have
        """
        # Filter events for unexisted logs
        # because of logs we have only from 6th month
        remove_outages_before = filter_otg_dates[0]
        remove_outages_after = filter_otg_dates[1]
        

        # we have only logs started from 2018-01-01
        df_outages = df_outages[(df_outages[self.otg_date_col] > remove_outages_before) & (
            df_outages[self.otg_date_col] < remove_outages_after)].reset_index(drop=True)
        print('Shape after filtering outages dates out of log range: ',
              df_outages.shape)

        """
        Clasifiying events types
        """
        print('Num of potentially Predictable outages: ',
              df_outages[df_outages[self.otg_cls_orig].isin(list_predictable_outages)].shape[0])
        print('Num of potentially UnPredictable outages:',
              df_outages[df_outages[self.otg_cls_orig].isin(list_random_outages)].shape[0])
        print('Num of unknown labeled outages:',
              df_outages[df_outages[self.otg_cls_orig].isin(list_unknown_outages)].shape[0])

        """
        Creating Labeled DataFrame for Training
        """
        df_lbl_outages = pd.DataFrame()

        # mv_lv_name_col='Alajaam'
        df_lbl_outages[self.otg_mv_lv_name_col] = df_outages[self.otg_mv_lv_name_col]

        # otg_date_col='Katkestuse algusaeg'
        df_lbl_outages[self.otg_date_col] = df_outages[self.otg_date_col]

        # cls_orig_col='Põhjus miks juhtus'
        df_lbl_outages[self.otg_cls_orig] = df_outages[self.otg_cls_orig].fillna(
            0)

        df_lbl_outages[self.otg_cls_orig]= df_outages[self.otg_cls_orig]
        df_lbl_outages[self.otg_cls_orig].replace(
            list_predictable_outages, 'Predictable', inplace=True)
        df_lbl_outages[self.otg_cls_orig].replace(
            list_synth_good, 'Predictable', inplace=True)  # Predictable as GOOD
        df_lbl_outages[self.otg_cls_orig].replace(
            list_random_outages, 'Random_UnPredictable', inplace=True)
        df_lbl_outages[self.otg_cls_orig].replace(
            list_unknown_outages, 'Unknown', inplace=True)

        df_lbl_outages['train_good']= df_outages[self.otg_cls_orig]
        df_lbl_outages['train_good'].replace(
            list_synth_good, 'Outage Not Expected', inplace=True)
        df_lbl_outages['train_good'].replace(
            list_random_outages, 'Random', inplace=True)
        df_lbl_outages['train_good'].replace(
            list_unknown_outages, 'Unknown', inplace=True)
        lst_outages= list(set(df_lbl_outages['train_good']).difference(
            ['Outage Not Expected', 'Unknown', 'Random']))
        df_lbl_outages['train_good'].replace(
            lst_outages, 'Outage Expected', inplace=True)

        """
        Removing 'Unknown' and 'Others' labels - for future purpose we can work with them
        """
        df_lbl_outages= df_lbl_outages[(df_lbl_outages.train_good != 'Unknown') & (
            df_lbl_outages.train_good != 'Random')].reset_index(drop=True)
        print('Shape after removal of Unknowns: ', df_lbl_outages.shape)

        """
        Removing time part from outage date-time
        """
        df_lbl_outages[self.otg_date_col]= df_lbl_outages[self.otg_date_col].dt.date  # Geting only date part from datetime

        def rm_spaces(df, columns):
            for col in columns:
                # Removing spaces from names as we did in dictionary
                df[col]= df[col].str.replace(' ', '')

            return df

        def get_dict_filtered_to_labels_mv_lv(df_dict, df_lbl_outages):

            df_dict= df_dict.drop(
                df_dict[df_dict[dev_level_up_1] == df_dict[dev_level_up_2]].index)

            ident= list(set(df_lbl_outages[self.otg_mv_lv_name_col]))

            df_dict= df_dict[(df_dict[dev_level_up_1].isin(ident)) | (
                df_dict[dev_level_up_2].isin(ident))]

            return df_dict

        df_dict= pd.read_feather(full_struct_dict_path)

        # Removing spaces in MV and LV names
        # Removing Spaces as we found misspellings with spaces in Labeled files
        df_dict= rm_spaces(
            df_dict, [dev_level_up_1, dev_level_up_2])
        df_lbl_outages= rm_spaces(df_lbl_outages, [self.otg_mv_lv_name_col])

        df_dict_filtered= get_dict_filtered_to_labels_mv_lv(
            df_dict, df_lbl_outages)

        df_lbl_outages['outage_on_MV']= df_lbl_outages[self.otg_mv_lv_name_col].isin(
            list(df_dict_filtered[dev_level_up_2]))
        df_lbl_outages['outage_on_LV']= df_lbl_outages[self.otg_mv_lv_name_col].isin(
            list(df_dict_filtered[dev_level_up_1]))

        """
        Removing Outages for LV or MV names not presented in logs
        """
        # Some data from labeled LM MV is not figured in Logs, because of misspelling
        df_lbl_outages= df_lbl_outages.drop(
            df_lbl_outages[df_lbl_outages['outage_on_MV'] == df_lbl_outages['outage_on_LV']].index).reset_index(drop=True)

        print('Shape after removal of outages for which devices not presented in logs: ',
              (df_lbl_outages.outage_on_MV.sum()+df_lbl_outages.outage_on_LV.sum()))

        # Adding Columns with either name of LV or MV, to use just one column, not filtering by two columns
        df_lbl_outages[dev_level_up_1]= df_lbl_outages.loc[df_lbl_outages[df_lbl_outages['outage_on_LV']
                                                                                 == True].index, self.otg_mv_lv_name_col]

        df_lbl_outages[dev_level_up_2]= df_lbl_outages.loc[df_lbl_outages[df_lbl_outages['outage_on_MV']
                                                                                 == True].index, self.otg_mv_lv_name_col]

        """
        Removing reccuring event in less then 7 days
        """
        # Drop duplicate occurance at same day
        df_lbl_outages= df_lbl_outages.drop_duplicates(
            subset=[self.otg_mv_lv_name_col, self.otg_date_col])

        # Sort by dates of outage
        df_lbl_outages= df_lbl_outages.sort_values(by=self.otg_date_col)

        # Filtering MV level
        # Get MVs names that have multiple outages in labels file
        mvs= []
        for dev in set(df_lbl_outages.mv_substation_name):
            if (df_lbl_outages[df_lbl_outages.mv_substation_name == dev].shape[0]) > 1:
                mvs.append(dev)

        dates_to_remove= pd.DataFrame(
            columns=[dev_level_up_2, self.otg_date_col])
        for dev in mvs[:]:
            dates= list(
                df_lbl_outages.loc[df_lbl_outages[df_lbl_outages.mv_substation_name == dev].index, self.otg_date_col])

            i= 0
            for date in dates:
                if (i+1) <= (len(dates)-1):
                    dist= dates[i+1]-date
                    if dist < np.timedelta64(8, 'D'):
                        dates_to_remove= dates_to_remove.append(
                            {dev_level_up_2: dev, self.otg_date_col: dates[i+1]}, ignore_index=True)
                i += 1

        # Removing recurring outages with distance less or equal to 7 days
        for _, row in dates_to_remove.iterrows():
            # print (row.mv_substation_name, row[self.otg_date_col])
            df_lbl_outages= df_lbl_outages.drop(df_lbl_outages[(df_lbl_outages.mv_substation_name == row.mv_substation_name) & (
                df_lbl_outages[self.otg_date_col] == row[self.otg_date_col])].index)

        # Filtering LV level
        # Get LVs names that have multiple outages in labels file
        lvs= []
        for dev in set(df_lbl_outages.lv_substation_name):
            if (df_lbl_outages[df_lbl_outages.lv_substation_name == dev].shape[0]) > 1:
                # print (dev, df_lbl_outages[df_lbl_outages.lv_substation_name==dev].shape[0])
                lvs.append(dev)

        dates_to_remove= pd.DataFrame(
            columns=[dev_level_up_1, self.otg_date_col])
        for dev in lvs[:]:
            dates= list(
                df_lbl_outages.loc[df_lbl_outages[df_lbl_outages.lv_substation_name == dev].index, self.otg_date_col])

            i= 0
            for date in dates:
                if (i+1) <= (len(dates)-1):
                    dist= dates[i+1]-date
                    if dist < np.timedelta64(8, 'D'):
                        # print (dev,dates,dist)
                        dates_to_remove= dates_to_remove.append(
                            {dev_level_up_1: dev, self.otg_date_col: dates[i+1]}, ignore_index=True)
                i += 1

        # Removing recurring outages with distance less or equal to 7 days
        for _, row in dates_to_remove.iterrows():
            # print (row.lv_substation_name, row[self.otg_date_col])
            df_lbl_outages= df_lbl_outages.drop(df_lbl_outages[(df_lbl_outages.lv_substation_name == row.lv_substation_name) & (
                df_lbl_outages[self.otg_date_col] == row[self.otg_date_col])].index)

        print('Shape after removal of recurring outages: ', df_lbl_outages.shape)

        # Saving labeled outages prepared tor training
        df_lbl_outages.to_csv(self.train_lbls_filename, index=False)
        # return df_lbl_outages
