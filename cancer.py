#!/usr/bin/python
# coding:utf-8
# Copyright (C) 2019-2020 All rights reserved.
# FILENAME:  cancer.py
# VERSION: 	 1.0
# CREATED: 	 2020-05-06 19:20
# AUTHOR: 	 Aekasitt Guruvanich <aekazitt@gmail.com>
# DESCRIPTION:
#
# HISTORY:
#*************************************************************
from math import sqrt
### Third-Party Packages ###
from numpy import log, square
from pandas import read_csv, cut
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
### Local Modules ###
from helpers.logger import Logger
from helpers.pickler import Pickler

def cancer_survival():
  ### Initiate Logger Instance ###
  logger = Logger.get_instance('cancer_survival')
  ### Initiate Pickler Instance ###
  pickler = Pickler.get_instance()
  data = pickler.load('radiomics_v2')
  if data is None:
    data = read_csv('data/radiomics_v2.csv')
    pickler.save(data, 'radiomics_v2')
  ### Rename Columns with Black Space to use Underscores instead
  renamed_cols = {
      col: '_'.join(col.split(' ')) for col in filter(lambda x: x[0]!='v', data.columns)
  }
  logger.info(renamed_cols)
  data.rename(columns=renamed_cols, inplace=True)
  logger.info(f'Test Data: {data.shape}')
  ### Checks for null values and find the percentage of null values that we have ###
  logger.info(f'Columns with Null Data:\n{data.isnull().any()}')
  logger.info(f'Percentage of Null Data:\n{data.isnull().sum() / data.shape[0]}')

  ### Column Descriptions ###
  # Time to Event is the amount of time from the date of data collection until a patient's day of death or until his/her last check-up
  # Patient Status is the patient's latest status (0 = alive, 1 = dead)
  # Patient Status at 3-Year is the patient's status at 3-year mark (0 = alive, 1 = dead, -1 = unknown)
  # v_n's are radiomics features extracted from ROI drawn by radiologists

  ### Drop Duplicates ###
  data.drop_duplicates(inplace=True)

  ### Clinical_C looks skewed, must be corrected using Square Values ###
  data['Clinical_C_Squared'] = square(data.Clinical_C)
  data.drop(['Clinical_C'], axis=1, inplace=True)

  ### Clinical_D seems to have an outlying tail in the positive ###
  ### Remove Outliers ###
  data = data[(data.Clinical_D < data.Clinical_D.quantile(.95)) & (data.Clinical_D > 0)]
  ### Still Skewed, try Logarithmic Function ###
  data['Clinical_D_Log'] = log(data.Clinical_D)
  ### Looks Good, now drop original Clinical_D column ###
  data.drop(['Clinical_D'], axis=1, inplace=True)
  
  ### Remove invalid rows from with 0 Age
  data = data[(data['Age'] > 0)]
  age_range = 3
  num_bins = int(data.Age.max() / age_range)
  data['Age_Range'] = cut(data['Age'], num_bins, labels=False)

  ### When Patient_Status_at_3_Year is unknown, impute with last known data ###
  data['Patient_Status_at_3_Year'] = data[['Patient_Status_at_3_Year', 'Patient_Status']].max(1)

  ### Train Model ###
  fold = 5
  forest_params = dict( \
    criterion= ['gini'], \
    max_features= ['auto', 'sqrt', 'log2'], \
    min_samples_split= range(2, 11), \
    n_estimators= range(10, 21) \
  )
  gsv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=forest_params, \
    scoring='neg_mean_absolute_error', verbose=10, n_jobs=15, cv=fold)

  trainable_data = data[data.Patient_Status_at_3_Year.notna()]
  x_train = trainable_data.drop(['Patient_ID', 'Time_to_Event', 'Patient_Status', 'Patient_Status_at_3_Year'], axis=1)
  y_train = trainable_data['Patient_Status_at_3_Year']
  y_stratify = trainable_data['Gender'] * y_train
  x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=.8, stratify=y_stratify)
  gsv.fit(x_train, y_train)
  model = gsv.best_estimator_
  logger.info(model)

  y_predict = model.predict(x_test)
  num_wrong_predictions = (y_predict != y_test).sum()
  r2 = r2_score(y_test, y_predict)
  mse = mean_squared_error(y_test, y_predict)
  rmse = sqrt(mse)
  logger.info(f'Number of Wrong Predictions: {num_wrong_predictions} / {len(y_predict)}')
  logger.info(f'R2: {r2:.4f}')
  logger.info(f'Mean Squared Error: {mse:.4f}')
  logger.info(f'Root Mean Squared Error: {rmse:.4f}')

  ### Save Model ###
  pickler.save(model, 'cancer_survival_estimator')

  Logger.release_instance()
  Pickler.release_instance()

if __name__ == '__main__':
  cancer_survival()
