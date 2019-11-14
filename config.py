# Setup of logs parser
mode = 'prod'
#mode = 'train'
#mode = 'prep_dict_lbls'

"""
Available modes: 
	prep_dict_lbls - prepare dictionary of devices structure and labeled file that needed as input for training
		req input: log file/s presented
		output: dictionary of structure to average events counts
				prepared outages labels for training model - removed unvaluable, reoccurring, etc
		
	train - train the model on logs files located in ./logs dir and save trained model and columns releted to model
		req input: 	log file/s presented, averaging dictionary presented, labeled outages presented
		output:		trained model to use on prod full thresholded logs
					columns name for trained model to subset prod full thresholded logs
		
	prod - predict outages based on logs files located in ./logs dir
		req input: 	log file/s presented, model file presented, model column names file presented, averaging dictionary presented, labeled outages presented
		output:		predictions and detailed predictions of outages for each givven log file separately
"""

# PREPARE CONF: Config for data preparation. Relevant for both Train and Prod methods

# Config dictionaries
full_struct_dict_path = './dict/avg_dict_full'

# Config for Preprocessing of Logs
# Config for getting list of files to proceed with
path = './logs'
extension_read = '.csv'
excl_subdirs = True

# Config list of columns to read from log file
cols_read = ['col1', 'col2', '...', 'colN'] # columns to read from log file

# Config descriptor for columns in logs
date_col = 'date column name here'
event_col = 'column name containing events here'
actual_log_collector_dev_col = 'column name where device names that generates logs here'
dev_agg_col = 'name of column here on which to aggregate, if need to use one or few more levels'
logs_cols_desc = [dev_agg_col, date_col,
                  event_col, actual_log_collector_dev_col]

# Config for averaging strategy if decided to group on level different from actual device that collects logs
avg_strategy = 'FullDictAvg'
"""
'NoAvg' = No Averaging
'DayAvg' = Averaging based just on daily number of unique MPs(EICs) that generated logs at this day
'CurrPerDictAvg' = Averaging based on dictionary created from current period logs (i.e. month, or 7 days)
'FullDictAvg' = Averaging based on full dictionary (Need to redo case when ONE MP(EIC) presented in TWO MVs in case of failure and switching)
"""

# Config list of important event types to subset dataset, to decrease number of feature for modeling
relevant_type_codes = ['1.11.111.111', '1.11.222.111', '2.11.111.111', '...']

# Config list of events to sum up, where it makes sense to sum all Phases (L1+L2+L3) events, and to decrease number of features for model
lst_sumup = [['Power down/up', '1.11.111.111', '1.11.111.112'],
             ['Overvoltage Phase *',
                 '1.11.222.111', '1.11.222.112', '1.11.222.113'],
             ['Voltage L* resume',
                 '2.11.111.111', '2.11.111.112', '2.11.111.113']]

# Config relevant for both Train and Prod methods
period = 7
interval = 7

# TRAIN CONF: Config particularly for Train
train_lbls_filename = './train/outages_info/train_labels_filtered.csv' # path where outages labeled and preperad file will generated at 'prep_dict_lbls' mode

otg_date_col = 'outage_date'  # internal column name
otg_cls_orig = 'outage_cls_orig'  # internal column name
otg_cls_model = 'outage_cls_model' # internal column name
otg_trn_good = 'train_good' # internal column name
otg_mv_lv_name_col='substation_name'  # internal column name
otg_affect_cust_col = 'affected_cust_col'  # internal column name

otg_cols_desc = [otg_date_col, otg_cls_orig,
                 otg_cls_model, otg_trn_good, otg_mv_lv_name_col, otg_affect_cust_col]

dev_level_up_1 = 'lv column name'   # LV - low voltage substation
dev_level_up_2 = 'mv column name'   # MV - low voltage substation
devs_cols = [actual_log_collector_dev_col, dev_level_up_1,
             dev_level_up_2]  # first should be actual log collector

"""
Clasifiying events types
"""
#  Problems in the system itself - behaviour of logs shoud be changed
list_predictable_outages = ['Amortization', '...']
# Things we cant predict (weather human)
list_random_outages = ['Vandalism', 'Theft', '...',]
# Unknown faults (need clarification by us and mapping to existed events)
import numpy as np
list_unknown_outages = [np.NaN, '...']
# Labels for Synthetic GOOD data - to train as Good device before outages - at least logs should be good before event happened
list_synth_good = ['Vandalism', 'Theft', '...']

"""
Configuring path and columns description for files containing historical outages info
"""
# Source Number 1:
filename = './train/outages_info/outages_2018.xlsx'
orgn_date_col = 'column name containing date of outage here'
orgn_affected_cust_col = 'column name containing number of affected customers if presented'
orgn_mv_lv_name_col = 'column name in historical outage csv file containing name of device for which outage took place'
orgn_cls_orig_col = 'column name in historical outage csv file containing class of outage labeled by human'
outages_files=[filename, orgn_date_col, orgn_affected_cust_col, orgn_mv_lv_name_col, orgn_cls_orig_col]
# Source Number 2:
filename = './train/outages_info/outages_2019.xlsx'
orgn_date_col = 'column name containing date of outage here'
orgn_affected_cust_col = 'column name containing number of affected customers if presented'
orgn_mv_lv_name_col = 'column name in historical outage csv file containing name of device for which outage took place'
orgn_cls_orig_col = 'column name in historical outage csv file containing class of outage labeled by human'
outages_files.append([filename, orgn_date_col, orgn_affected_cust_col, orgn_mv_lv_name_col, orgn_cls_orig_col])

# Remove outages for training depending on logs presented
remove_outages_before = '2018-01-10 00:00:00'
remove_outages_after = '2019-07-01 00:00:00'
filter_otg_dates=[remove_outages_before, remove_outages_after]


# PROD CONF: Config particularly for Prod
prod_model_filename = './models/finalized_outages_model.sav' # where to put trained model
