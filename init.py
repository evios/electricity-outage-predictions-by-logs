!pip install numpy
!pip install pandas
!pip install scipy
!pip install sklearn 
!pip install xgboost 


import pandas as pd

import config as cfg
from logs_preprocessing import preproc
from prototype_production import prod
from train import train


# -- START -- Preparing of Logs
logs_prep = preproc.logs_prep(full_struct_dict_path=cfg.full_struct_dict_path)

filenames = logs_prep.get_files_list(
    path=cfg.path, extension_read=cfg.extension_read, excl_subdirs=cfg.excl_subdirs)

for file in filenames:
    # Logs Reading
    df = logs_prep.csv_read_and_save_to_feather(
        file, cols_read=cfg.cols_read, date_col=cfg.date_col, create_feather=False)  # Read logs from CSV

    # Removing spaces in devices names # Showed it here as important to see that all spaces are removed during process preparation,
    # hence affects all process ahead and external data sources should be treat in the same way
    df = logs_prep.rm_spaces(df, [cfg.dev_agg_col])

    # Logs grouping daily
    df_grouped, error = logs_prep.generate_dev_date_df(
        df, logs_cols_desc=cfg.logs_cols_desc, avg_strategy=cfg.avg_strategy)

    # Filtering unvaluable events types from defined list of events types
    df_grouped = logs_prep.filter_relevant_events(
        df_grouped, relevant_type_codes=cfg.relevant_type_codes)

    # Sum up same events for Phase 1, 2, 3
    df_grouped = logs_prep.sum_cols(df_grouped, cfg.lst_sumup)
    # -- END -- Preparing of Logs


    # -- !!! START !!! -- PRODUCTION BRANCH
    if cfg.mode == 'prod':

        prod = prod.prod_predict(cfg.prod_model_filename)

        # Suming up events X days before evaluation date
        df_model, error = prod.make_hist_df(
            df_grouped, period=cfg.period, interval=cfg.interval, logs_cols_desc=cfg.logs_cols_desc)

        # Thresholding outliers
        df_model_thresholded = prod.threshold_df_outliers_pre_model(df_model)

        # Predicting and thresholding most confident predictions in number of recurring cycles (iterations)
        df_preds = prod.cyclical_pred(
            df_model_thresholded, thresh=0.95, enforce_first_cycle=True, norm_trfm=True, skew_trfm=False)

        # Saving predictions to file
        prod.write_preds(df_preds, file)
    # -- !!! END !!! -- PRODUCTION BRANCH


    # -- !!! START !!! -- TRAINING BRANCH
    elif cfg.mode == 'train':

        trn = train.train_model(train_lbls_filename=cfg.train_lbls_filename,
                                otg_cols_desc=cfg.otg_cols_desc, logs_cols_desc=cfg.logs_cols_desc)

        # Geting labeled dataframe
        lst_lv_outages = trn.get_train_lbl(get_unq_devs=True)

        # getting subset for Train LV for concatenetaion all DFs in one DF containing only devices for which outages occured
        df = df[df[cfg.dev_agg_col].isin(lst_lv_outages)]
        dfs = dfs + tuple((df,))

if cfg.mode == 'train':
    # Concat all logs in one DF
    df = pd.concat(dfs, axis=0, sort=False, ignore_index=True).fillna(0)

    # Grouping all logs in one to prepare train dataframe
    df_model = trn.make_hist_df_train(
        df, period=cfg.period, interval=cfg.interval)

    # Get X and y DFs
    X, y = trn.prep_train_df_model(
        df_model, drp_low_var_ft=True, norm_trfm=True, skew_trfm=True)

    # Train and save model
    trn.train_save_modeling(X, y, cfg.prod_model_filename)
# -- !!! END !!! -- TRAINING BRANCH



# -- !!! START !!! -- LABELS AND DICTIONARY BUILDING
if cfg.mode == 'prep_dict_lbls':
    trn = train.train_model(train_lbls_filename=cfg.train_lbls_filename,
                            otg_cols_desc=cfg.otg_cols_desc, logs_cols_desc=cfg.logs_cols_desc)

    # Create Dictionary for averaging in train and production
    trn.build_dict(filenames, cfg.devs_cols,
                   filename_dict_to_wr=cfg.full_struct_dict_path)

    # Create and save labeled file for outage where we can take dependant variable 'y'
    trn.prep_labels(full_struct_dict_path=cfg.full_struct_dict_path, list_predictable_outages=cfg.list_predictable_outages, list_random_outages=cfg.list_random_outages,
                    list_unknown_outages=cfg.list_unknown_outages, list_synth_good=cfg.list_synth_good, outages_files=cfg.outages_files, filter_otg_dates=cfg.filter_otg_dates)
# -- !!! END !!! -- LABELS AND DICTIONARY BUILDING