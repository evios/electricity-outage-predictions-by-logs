# electricity-outage-predictions-by-logs
Electrical substations outage predictions by logs
----------------------------------------------------------------------------------------------------------------------
	Current solution is proof of concept and is first pilot stage to predict real outages based on logs from end custumer devices.
	Takes logs as input and return most confident predictions of outages written to file ./log-filename_result.xls

	Solution contains 3 main blocks (mirrored in working modes):
	1. Preparing dictionaries and outage labels to start training model (mode = prep_dict_lbls)
	2. Training model, with logs preprocessing (mode = train)
	3. Predicting outages for given log files  (mode = prod)

	Period of solution development: 01.11.2019 - 15.11.2019
----------------------------------------------------------------------------------------------------------------------
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
----------------------------------------------------------------------------------------------------------------------
	Instalation before running:
	In order to prepare environment only python installed is needed
	Python 3.8 (either installed as independantly or through some DS packages such as Anaconda, etc)
		Python last version Windows version https://www.python.org/downloads/release/python-380/

    
	Module names should be installed automatically while running 'init.py' using '!pip install <mod_name>'
		Modules can be installed also manually using './requirements.txt' file
----------------------------------------------------------------------------------------------------------------------
	Running:
		1. Put logs to ./logs dir
		2. Run init.py
----------------------------------------------------------------------------------------------------------------------
	Configuration:
		All configuration located in ./config.py
		Changing mode, dir structure, fields and columns names in config file
----------------------------------------------------------------------------------------------------------------------
	Directories structure:
	./                          # root directory
	.config.py                  # config for current run
	.init.py                    # initialization, start script
	.README						# Current info file 
	./requirements.txt			# Required for running solution modules. Should be installed automatically inside scripts
	|-> dict/                   # dictionaries
	|-> logs/                   # logs file/s - put here log file/s you want to predict outages for
	    |->     events_01-2018_v1.csv                       # log file
	|-> logs_preprocessing/     # logs preparation methods
	    |->     preproc.py                       # class with methods same for both Train and Prod
	|-> model/                  # trained model for prediction
	    |->     finalized_outages_model.sav                 # model
	    |->     finalized_outages_model_columns.csv         # columns names for model
	|-> prototype_production/   # live related methods
	|-> tmp/                    # directory to save intermediate results if flag is set to save feather
	    |->     logs/           # will save here log file/s created at intermediaries steps
	|-> train/                  # train related methods
----------------------------------------------------------------------------------------------------------------------
	Description of solution Step by Step
	You can start from any step and you can skip steps

	Logs Preparation for modeling
	Step 1: get list of files from given logs directory
	Step 2: read file by file (for cycle)
	    Step 3: group one file
		Step 3.1: save file if flag is true (using just to convert list of logs files ready for modeling)

	Mode: possible option is 'train' or 'prod'
	Mode = train
	    Step 4: filter file by labeled outages file (leave only devices NAME/ID that are in historical outages)
	    Step 5: concatenate (combine) to new - combined logs file
	Step 6: sum events for last X (7 in current workflow) days before outage
	    Step 6.1: save file if flag is true
	Step 7: train model 
	Step 8: save model and model columns names (features names)

	Mode = prod
	    Step 4: sum events for last X (7 in current workflow) days before last day in current file
		Step 4.1: save file if flag is true
	    Step 5: threshold outliers
	    Step 6: predict (in few cycles if set). Predict outages for period starting from last day in current log file +1d
	    Step 7: threshold most confident predictions
	    Step 8: save results to CSV/XLS
----------------------------------------------------------------------------------------------------------------------
