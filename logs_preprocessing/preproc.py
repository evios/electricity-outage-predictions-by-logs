import os
import re

import numpy as np
import pandas as pd


class logs_prep():
    def __init__(self, full_struct_dict_path):
        # Setup of logs parser
        # All config variables stored in ./config.py
        # Init variables
        self.full_struct_dict_path = full_struct_dict_path
        

    def get_files_list(self, path='../logs', extension_read='csv', excl_subdirs=True):
        """
        Return files containing extension as a part of filename (either in extension part or in filename part)

        Input:
        path = Path where to list files
        extension_read = extension or part of filename as pattern (i.e. 'csv' or '_feather_step1')
        excl_subdirs = Find files only in defined directory, or also include sub directories
        """
        self.path = path
        self.extension_read = extension_read
        self.excl_subdirs = excl_subdirs

        filenames = []
        # r=root, d=directories, f = files
        for r, _, f in os.walk(self.path):
            for file in f:
                if self.extension_read in file:
                    filenames.append(os.path.join(r, file))
            if self.excl_subdirs == True:  # go through only current directory, excluding sub-directories
                break

        return filenames

    def csv_read_and_save_to_feather(self, filename, cols_read, date_col, create_feather=False, dir_to_save='tmp/'): #, nrows=100):
        """
        Read CSV format and save it to Feather to improve performance if read of files needed more that once
        If log file is to big to save in feather it will save in chunks

        Input:
        filename = str, CSV log file
        cols_read = list, columns to read
        date_col = str, column to parse as date in pd.read_csv
        dir_to_save = str, path where to save feather if chosen create_feather = True
        create_feather = boolean, flag to save or not readed CSV file as Feather
        """
        self.cols_read = cols_read
        self.date_col = date_col
        
        print ('Reading file: ', filename)

        df = pd.read_csv(filename,
                         usecols=self.cols_read,
                         parse_dates=[self.date_col]) #, nrows=nrows)  # for debug purposes
        
        print ('File read succesfully!')
        
        print('Percentage of Missed Dates: ', round(
            df[df[self.date_col].isna()].shape[0]/df.shape[0]*100, 2))

        # Converting TimeZone to UTC and removing Offset (timezone)
        df[self.date_col] = pd.to_datetime(df[self.date_col], utc=True)
        df[self.date_col] = pd.DatetimeIndex(
            df[self.date_col]).tz_convert(None)

        print('Percentage of Missed Dates: ', round(
            df[df[self.date_col].isna()].shape[0]/df.shape[0]*100, 2))

        if create_feather == True:
            os.makedirs(dir_to_save, exist_ok=True)
            # Dividing DF into few smaller parts if too big, because error will arrised for too big DF:
            # Column 'reason' exceeds 2GB maximum capacity of a Feather binary column.
            max_rows = 30000000  # 30M rows, seems OK for this type of data in columns

            # threshold - if DF more than 20% biger than max_rows = write in chunks
            if df.shape[0] > (max_rows*1.2):
                chunks = df.shape[0]//max_rows+1

                # Write to Feather file in chunks
                for chunk_nmbr in range(chunks):
                    df[chunk_nmbr*max_rows:(chunk_nmbr+1)*max_rows].reset_index(
                        drop=True).to_feather(dir_to_save+filename+'_part_'+str(chunk_nmbr)+'_feather')
            else:
                df.to_feather(dir_to_save+filename+'_feather')

        return df

    def generate_dev_date_df(self, df, logs_cols_desc, avg_strategy='NoAvg'):
        """
        Input DF
        Will convert logs from format
        event_id        |   event               |   device_name         |   date
        1.11.111.111    |   Voltage L2 resume   |   12EEE-12345678-1	|   2019-04-03 07:42:23+03:00
        1.11.222.111    |   Undervoltage L2     |   12EEE-12345678-1	|   2019-04-03 08:03:18+03:00
        2.11.111.111    |   Undervoltage L1     |   12EEE-12345678-1	|   2019-04-03 02:58:23+03:00

        Ouput DF
        to (grouped by day and eic and summurize number of particular events of each type in logs)
        device_name			|   date        |   1.11.111.111    |   1.11.222.111	|   2.11.111.111
        12EEE-12345678-1	|   2019-04-03	|   10              |   12              |   58

        Input: 
            df = logs dataframe with columns of MP, LV, MV, event, date
            logs_cols_desc = pointers to columns name: 
                [0]= which column use to aggregate on
                [1]= which column use as date of event
                [2]= which column use as events column
                [3]= which column use as actual device that generated logs
            avg_strategy = strategy of averaging if device to aggregate on is not log collector device (i.e. substation)
        """

        test_error = 0  # for test purposes
        #dev_agg_col = logs_cols_desc[0]
        self.dev_agg_col = logs_cols_desc[0]
        self.date_col = logs_cols_desc[1]
        event_col = logs_cols_desc[2]
        actual_log_collector_dev_col = logs_cols_desc[3]

        # some events missing Metering point info
        # drop rows for absent Log Collector Device Name values
        try: df.dropna(subset=[actual_log_collector_dev_col], inplace=True)
        except: print('No NA values in column: ',actual_log_collector_dev_col, 'Proceeding' )

        # Geting only date part from datetime
        df[self.date_col] = df[self.date_col].dt.date # this will convert column from datetime64[ns] to object 
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        if self.dev_agg_col == actual_log_collector_dev_col:
            avg_strategy = 'NoAvg'
            print('Averagind not needed as device is Log Collector')

        # print ('Starting Grouping current file with strategy: ', avg_strategy)

        """ Averaging Strategy
        'NoAvg' = No Averaging
        'DayAvg' = Averaging based just on daily number of unique MPs(EICs) that generated logs at this day
        'CurrPerDictAvg' = Averaging based on dictionary created from current period logs (i.e. month, or 7 days)
        'FullDictAvg' = Averaging based on full dictionary (Need to redo case when ONE MP(EIC) presented in TWO MVs in case of failure and switching)
        """
        # print ('Avg strategy is: ', avg_strategy)
        
        if avg_strategy == 'NoAvg':  # NoAvg = No Averaging
            df_grouped = df[[self.dev_agg_col, self.date_col, event_col]].pivot_table(
                index=[self.dev_agg_col, self.date_col], columns=[event_col], aggfunc=len, fill_value=0)

            # For test Input==Output
            cnt_evnts = df.shape[0]
            # Passing test
            cnt_evnts_after_grouping = df_grouped.values.sum()
            if cnt_evnts == cnt_evnts_after_grouping:
                print('Test Passed: Data Transormed Correctly, Totally:',
                      cnt_evnts, ', No events were lost for given EICs')
            else:
                test_error = [cnt_evnts, cnt_evnts_after_grouping]

            df_grouped = df_grouped.reset_index()

        # FullDictAvg = Averaging based on full dictionary (Need to redo case when ONE MP(EIC) presented in TWO MVs in case of failure and switching)
        elif avg_strategy == 'FullDictAvg':
            # df.columns.difference(['up_name', 'created_at', 'type_code']) # get all columns not in this list
            df_grouped = df[[self.dev_agg_col, self.date_col, event_col]].pivot_table(
                index=[self.dev_agg_col, self.date_col], columns=[event_col], aggfunc=len, fill_value=0)
            df_grouped = self.full_dict_avg(df_grouped, logs_cols_desc)
            df_grouped[self.date_col] = pd.to_datetime(df_grouped[self.date_col])
            
        # DayAvg = Averaging based just on daily number of unique MPs(EICs) that generated logs at this day
        elif avg_strategy == 'DayAvg':
            df_grouped = df[[self.dev_agg_col, self.date_col, event_col, actual_log_collector_dev_col]].pivot_table(
                index=[self.dev_agg_col, self.date_col], columns=[event_col], aggfunc=(lambda x: len(x)/len(set(x))), fill_value=0)
            df_grouped.columns = df_grouped.columns.get_level_values(1)
            df_grouped = df_grouped.reset_index()

        # CurrPerDictAvg = Averaging based on dictionary created from current period logs (i.e. month, or 7 days)
        elif avg_strategy == 'CurrPerDictAvg':
            print('Nothing here yet, Lower Priority')

        else:
            print(
                'Wrong Averaging Strategy Choosen. Please use description to choose right one!')
            test_error = 'avg_strategy variable error'

        # print ('Current file grouped with strategy: ', avg_strategy)

        return df_grouped, test_error

    def full_dict_avg(self, df, logs_cols_desc):
        """
        Averaging events by dividing on full dictionary created externally 
        Dictionary represents network structure of devices to clarify what number of devices on what level
        """
        dev_agg_col = logs_cols_desc[0]
        date_col = logs_cols_desc[1]

        df_dict = pd.read_feather(self.full_struct_dict_path)

        # define column name to look for number of devices
        num_of_devs_on_agg_dev_col = 'eics_on_'+dev_agg_col[:2]
        #num_of_devs_on_agg_dev_col = 'log_devs_on_'+dev_agg_col[:2]

        # aggregate dictionary fod current aggregeted device level (MV or LV)
        df_eics_on_agg_dev = df_dict.groupby(dev_agg_col).first().reset_index()[
            [dev_agg_col, num_of_devs_on_agg_dev_col]]

        # adding columns with number of lowest level MPs(EICs)(Log collectore device) on current aggregatable device
        df = pd.merge(df.reset_index(), df_eics_on_agg_dev,
                      how='left', on=dev_agg_col)

        # Averaging (dividing each column to number of devices)
        df_avg = df.drop(columns=[dev_agg_col, num_of_devs_on_agg_dev_col, date_col]).transform(
            lambda x: x / df[num_of_devs_on_agg_dev_col])

        # Combining back
        df = pd.concat([df[[dev_agg_col, date_col]], df_avg],
                       sort=False, axis=1)

        return df

    def curr_per_dict_avg(self, df):
        """
        Lower Priority - ToDo in Future
        """
        return df

    def filter_relevant_events(self, df, relevant_type_codes):
        """
        Subsetting DF to important event types only, to decrease number of feature for modeling
        Filtering out not important events
        """
        relevant_type_codes.extend([self.dev_agg_col, self.date_col])
        df = df.drop(columns=df.columns.difference(relevant_type_codes))

        return df

    def sum_cols(self, df, lst_sumup):
        """
        Sum up same events for Phase 1, 2, 3
        Sum up particular events, where it makes sense to sum all Phases (L1+L2+L3) events, and to decrease number of features for model
        """
        try:
            for related_events in lst_sumup:
                df[related_events[0]]=df[related_events[1:]].sum(axis=1)
                df=df.drop(columns=related_events[1:])
            print('Dataset Succesfully summed up')
        except:
            print('Dataset has already been summed up')

        return df

    def rm_spaces(self, df, columns):
        """
        Removing all spaces in all rows, indexes of given columns
        """
        for col in columns:
            df[col]=df[col].str.replace(' ', '') # Removing spaces from names as we did in dictionary
        return df
