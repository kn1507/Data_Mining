#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:39:41 2020

@author: duygu86
"""

import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import scipy.stats as stats

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file
from itertools import tee, islice, chain
from datetime import datetime
from tqdm import tqdm
import logging
from os import path

import random
def randomiseMissingData(df2):
    "randomise missing data for DataFrame (within a column)"
    df = df2.copy()
    for col in df.columns:
        data = df['prop_review_score']
        mask = data.isnull()
        samples = random.choices( data[~mask].values , k = mask.sum() )
        data[mask] = samples
    return df

# create logger with 'spam_application'
logger = logging.getLogger('assignment2')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('train_test.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

def log_evaluation(period=1, show_stdv=True):
    """Create a callback that logs evaluation result with logger.

    Parameters
    ----------
    period : int
        The period to log the evaluation results

    show_stdv : bool, optional
         Whether show stdv if provided

    Returns
    -------
    callback : function
        A callback that logs evaluation every period iterations into logger.
    """

    def _fmt_metric(value, show_stdv=True):
        """format metric string"""
        if len(value) == 2:
            return '%s:%g' % (value[0], value[1])
        elif len(value) == 3:
            if show_stdv:
                return '%s:%g+%g' % (value[0], value[1], value[2])
            else:
                return '%s:%g' % (value[0], value[1])
        else:
            raise ValueError("wrong metric value")

    def callback(env):
        if env.rank != 0 or len(env.evaluation_result_list) == 0 or period is False:
            return
        i = env.iteration
        if i % period == 0 or i + 1 == env.begin_iteration or i + 1 == env.end_iteration:
            msg = '\t'.join([_fmt_metric(x, show_stdv) for x in env.evaluation_result_list])
            logger.info('[%d]\t%s\n' % (i, msg))

    return callback

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

if not path.exists("./expedia.pkl") or not path.exists("./exptest.pkl"):
    expedia_df = pd.read_csv('training_set_VU_DM.csv')
    exptest_df    = pd.read_csv('test_set_VU_DM.csv')
    
    logger.info('Train and Test Data Read!')
    logger.info('')
    
    # most_booked_hotel = pd.DataFrame(expedia_df.groupby('prop_id')['booking_bool'].sum().reset_index())
    # most_booked_hotel.columns = ['prop_id', 'booking_count']
    # most_booked_hotel.sort_values(by='booking_count', ascending=False)
    
    # expedia_df = pd.merge(expedia_df, most_booked_hotel, how='left', left_on='prop_id', right_on='prop_id')
    
    srch_crit = ['date_time', 'srch_destination_id','srch_length_of_stay', 'srch_booking_window',
                 'srch_adults_count','srch_children_count','srch_room_count','srch_saturday_night_bool']
    hotel_static = ['prop_id', 'prop_country_id','prop_starrating','prop_review_score', 
                    'prop_brand_bool','prop_location_score1','prop_location_score2',
                    'prop_log_historical_price']
    hotel_dyn = ['position','price_usd','promotion_flag','srch_query_affinity_score',
                 'click_bool','booking_bool','gross_bookings_usd','prop_log_historical_price'] 
    visitor_loc = ['visitor_location_country_id','orig_destination_distance']
    visitor_agg = ['visitor_hist_starrating','visitor_hist_adr_usd']
    competitive = ['comp1_rate', 'comp1_inv','comp1_rate_percent_diff', 
                   'comp2_rate', 'comp2_inv','comp2_rate_percent_diff', 
                   'comp3_rate', 'comp3_inv','comp3_rate_percent_diff', 
                   'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 
                   'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 
                   'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 
                   'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 
                   'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']
    other_info = ['srch_id','site_id','random_bool']
    
    cases = [srch_crit,hotel_static,hotel_dyn,visitor_loc,visitor_agg,competitive,other_info]
    # preview the data
    expedia_df.head()
    
    expedia_df.describe()
    
    logger.info('Start Plotting...')
    
    fig, (axis1,axis2) = plt.subplots(2,1,figsize=(15,10))
    
    bookings_df = expedia_df[expedia_df["booking_bool"] == 1]
    
    # What are the most countries the customer travel from?
    sns.countplot('visitor_location_country_id',data=bookings_df.sort_values(by=['visitor_location_country_id']),ax=axis1,palette="Set3")
    
    # What are the most countries the customer travel to?
    sns.countplot('prop_country_id',data=bookings_df.sort_values(by=['prop_country_id']),ax=axis2,palette="Set3")
    
    # Combine both plots
    # fig, (axis1) = plt.subplots(1,1,figsize=(15,5))
    
    # sns.distplot(bookings_df["hotel_country"], kde=False, rug=False, bins=25, ax=axis1)
    # sns.distplot(bookings_df["user_location_country"], kde=False, rug=False, bins=25, ax=axis1)
    
    # Where do most of the customers from a country travel?
    user_country_id = 219
    
    country_customers = expedia_df[expedia_df["visitor_location_country_id"] == user_country_id]
    country_customers["prop_country_id"].value_counts().plot(kind='bar',colormap="Set3",figsize=(15,5))
    plt.show()
    
    # expedia_df['Date']  = expedia_df['date_time'].apply(lambda x: (str(x)[:7]) if x == x else np.nan)
    # date_bookings  = expedia_df.groupby('Date')["booking_bool"].sum()
    
    # ax1 = date_bookings.plot(legend=True,marker='o',title="Total Bookings", figsize=(15,5)) 
    # ax1.set_xticks(range(len(date_bookings)))
    # xlabels = ax1.set_xticklabels(date_bookings.index.tolist(), rotation=90)
    
    # fig, (axis1) = plt.subplots(1,1,figsize=(15,3))
    
    # ax2 = sns.boxplot([date_bookings.values], whis=np.inf,ax=axis1)
    # ax2.set_title('Important Values')
    
    for case in cases:
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
        sns.heatmap(expedia_df[case].corr(),cmap='coolwarm',ax=ax,annot=True,linewidths=2)
        plt.show()
    
        
    missing_data = 100*(1-expedia_df.isnull().sum(axis=0)/expedia_df.shape[0])
    missing_data.sort_values().plot(kind='bar',colormap="Set3",figsize=(15,5))
    plt.show()
    
    logger.info('Plotting ends...')
    logger.info('Preprocessing...')
    
    mms = preprocessing.MinMaxScaler()
    
    expedia_df.drop(columns=['date_time'],inplace=True)
    expedia_df = randomiseMissingData(expedia_df)
    expedia_df['prop_location_score2'].fillna((expedia_df['prop_location_score2'].mean()), inplace=True)
    expedia_df['orig_destination_distance'].fillna((expedia_df['orig_destination_distance'].median()), inplace=True)
    expedia_df[['orig_destination_distance']] = mms.fit_transform(expedia_df[['orig_destination_distance']])
    expedia_df = expedia_df.fillna(-2)
    
    th = 1
    expedia_df['price_usd'].loc[expedia_df['price_usd']<th] = th
    expedia_df['price_usd'] = expedia_df['price_usd'].map(np.log10)
    
    exptest_df.drop(columns=['date_time'],inplace=True)
    exptest_df = randomiseMissingData(exptest_df)
    exptest_df['prop_location_score2'].fillna((exptest_df['prop_location_score2'].mean()), inplace=True)
    exptest_df['orig_destination_distance'].fillna((exptest_df['orig_destination_distance'].median()), inplace=True)
    exptest_df[['orig_destination_distance']] = mms.fit_transform(exptest_df[['orig_destination_distance']])
    
    exptest_df = exptest_df.fillna(-2)
    exptest_df['price_usd'].loc[exptest_df['price_usd']<th] = th
    
    exptest_df['price_usd'] = exptest_df['price_usd'].map(np.log10)
    
    logger.info('Preprocessing done...')
    
    # expedia_df['prop_review_score'] = expedia_df['prop_review_score'].fillna(0)
    # expedia_df['prop_location_score2'] = expedia_df['prop_location_score2'].fillna(0)
    # expedia_df['srch_query_affinity_score'] = expedia_df['srch_query_affinity_score'].fillna(expedia_df['srch_query_affinity_score'].min())
    num_feats = ['prop_review_score','prop_location_score1','prop_location_score2',
                 'prop_log_historical_price','price_usd','srch_query_affinity_score',
                 'orig_destination_distance']
    
    logger.info('Additional Feature generation...')
    
    expedia_df['price_diff_from_recent'] = expedia_df['price_usd']-expedia_df['prop_log_historical_price']
    exptest_df['price_diff_from_recent'] = exptest_df['price_usd']-exptest_df['prop_log_historical_price']
    
    
    expedia_df['price_order'] = expedia_df.groupby('srch_id')['price_usd'].apply(lambda x : np.argsort(x))
    
    exptest_df['price_order'] = exptest_df.groupby('srch_id')['price_usd'].apply(lambda x : np.argsort(x))
    
    # for idx in expedia_df['srch_id'].unique():
    #     idxs = expedia_df['srch_id'] == idx
    #     expedia_df['price_order'][idxs] = expedia_df['price_usd'][idxs].argsort()
    
    # exptest_df['price_diff_from_recent'] = exptest_df['price_usd']-exptest_df['prop_log_historical_price']
    # exptest_df['price_order'] = exptest_df['srch_id'].copy()
    # for idx in exptest_df['srch_id'].unique():
    #     idxs = exptest_df['srch_id'] == idx
    #     exptest_df['price_order'][idxs] = exptest_df['price_usd'][idxs].argsort()
    
    logger.info('Grouped feature generation...')
    logger.info('Numerical features:'+str(num_feats))
    
    for ff in num_feats:    
        dummy = pd.DataFrame(expedia_df.groupby('srch_id')[ff].mean().reset_index())
        dummy.columns = ['srch_id', 'mean_srch_'+ff]
        dummy.sort_values(by='mean_srch_'+ff, ascending=False)
        expedia_df = pd.merge(expedia_df, dummy, how='left', left_on='srch_id', right_on='srch_id')
    
        dummy = pd.DataFrame(expedia_df.groupby('prop_id')[ff].mean().reset_index())
        dummy.columns = ['prop_id', 'mean_prop_'+ff]
        dummy.sort_values(by='mean_prop_'+ff, ascending=False)
        expedia_df = pd.merge(expedia_df, dummy, how='left', left_on='prop_id', right_on='prop_id')
    
        dummy = pd.DataFrame(expedia_df.groupby('srch_destination_id')[ff].mean().reset_index())
        dummy.columns = ['srch_destination_id', 'mean_dest_'+ff]
        dummy.sort_values(by='mean_dest_'+ff, ascending=False)
        expedia_df = pd.merge(expedia_df, dummy, how='left', left_on='srch_destination_id', right_on='srch_destination_id')
    
        dummy = pd.DataFrame(exptest_df.groupby('srch_id')[ff].mean().reset_index())
        dummy.columns = ['srch_id', 'mean_srch_'+ff]
        dummy.sort_values(by='mean_srch_'+ff, ascending=False)
        exptest_df = pd.merge(exptest_df, dummy, how='left', left_on='srch_id', right_on='srch_id')
    
        dummy = pd.DataFrame(exptest_df.groupby('prop_id')[ff].mean().reset_index())
        dummy.columns = ['prop_id', 'mean_prop_'+ff]
        dummy.sort_values(by='mean_prop_'+ff, ascending=False)
        exptest_df = pd.merge(exptest_df, dummy, how='left', left_on='prop_id', right_on='prop_id')
    
        dummy = pd.DataFrame(exptest_df.groupby('srch_destination_id')[ff].mean().reset_index())
        dummy.columns = ['srch_destination_id', 'mean_dest_'+ff]
        dummy.sort_values(by='mean_dest_'+ff, ascending=False)
        exptest_df = pd.merge(exptest_df, dummy, how='left', left_on='srch_destination_id', right_on='srch_destination_id')
    
        print(ff,'is done!')
        logger.info(ff+' is done...')
    
    expedia_df.to_pickle("./expedia.pkl")
    exptest_df.to_pickle("./exptest.pkl")

expedia_df = pd.read_pickle("./expedia.pkl")
exptest_df = pd.read_pickle("./exptest.pkl")
# for ff in num_feats:
#     expedia_df['mean_srch_'+ff] = 0.0
#     for idx in expedia_df['srch_id'].unique():
#         idxs = expedia_df['prop_id'] == idx
#         expedia_df['mean_srch_'+ff][idxs] = expedia_df[ff][idxs].mean()

# for ff in num_feats:
#     expedia_df['mean_prop_'+ff] = 0.0
#     for idx in expedia_df['prop_id'].unique():
#         idxs = expedia_df['prop_id'] == idx
#         expedia_df['mean_prop_'+ff][idxs] = expedia_df[ff][idxs].mean()

# for ff in num_feats:    
#     expedia_df['mean_dest_'+ff] = 0.0
#     for idx in expedia_df['srch_destination_id'].unique():
#         idxs = expedia_df['srch_destination_id'] == idx
#         expedia_df['mean_dest_'+ff][idxs] = expedia_df[ff][idxs].mean()
 
logger.info('Training, test and validation data separation...')

params = {'objective': 'rank:ndcg',
          'nthread':-1, 
          'silent': False,
          'max_depth': 9,
          'learning_rate': 0.13877466758976015,
          'colsample_bytree': 0.6,
          'n_estimators': 100}

callbacks = [log_evaluation(1, True)]
num_round = 1000
# separate booking bools and keep all the booking_bools
booking_bool_0_df = expedia_df[expedia_df['booking_bool'] == 0]
booking_bool_1_df = expedia_df[expedia_df['booking_bool'] == 1] 
unique_srch_ids_1 = booking_bool_1_df['srch_id'].unique().tolist()

### Train and Validation set ###
train_ids, validation_ids = train_test_split(unique_srch_ids_1, test_size=0.25)
book_1_train = booking_bool_1_df[booking_bool_1_df['srch_id'].isin(train_ids)]
book_1_validation = booking_bool_1_df[booking_bool_1_df['srch_id'].isin(validation_ids)]

book_0_train = booking_bool_0_df[booking_bool_0_df['srch_id'].isin(train_ids)]
book_0_validation = booking_bool_0_df[booking_bool_0_df['srch_id'].isin(validation_ids)]

train_df = pd.concat([book_0_train, book_1_train])
validation_df = pd.concat([book_0_validation, book_1_validation])

test_df = exptest_df

# sort train_df by srch_id #
train_df.sort_values(by='srch_id', inplace=True)
# get corresponding srch_id counts
srch_counts_train_df = pd.DataFrame(train_df['srch_id'].value_counts().reset_index())
srch_counts_train_df.columns = ['srch_id', 'count']
srch_counts_train_df.sort_values(by='srch_id', inplace=True)
group_counts_train = srch_counts_train_df['count'].to_numpy()

# for validation set
validation_df.sort_values(by='srch_id', inplace=True)
# get corresponding srch_id counts
srch_counts_val_df = pd.DataFrame(validation_df['srch_id'].value_counts().reset_index())
srch_counts_val_df.columns = ['srch_id', 'count']
srch_counts_val_df.sort_values(by='srch_id', inplace=True)
group_counts_val = srch_counts_val_df['count'].to_numpy()

query_id_train = train_df['srch_id'].to_numpy()
prop_id_train = train_df['prop_id'].to_numpy()
y_train = train_df['click_bool'].to_numpy()
train_df.drop(columns=['srch_id','prop_id','click_bool','booking_bool',
                       'gross_bookings_usd','position'], inplace=True)
X_train = train_df.to_numpy()

query_id_val = validation_df['srch_id'].to_numpy()
prop_id_val = validation_df['prop_id'].to_numpy()
y_val = validation_df['click_bool'].to_numpy()
validation_df.drop(columns=['srch_id','prop_id','click_bool','booking_bool',
                            'gross_bookings_usd','position'], inplace=True)
X_val = validation_df.to_numpy()

srch_counts_test_df = pd.DataFrame(test_df['srch_id'].value_counts().reset_index())
srch_counts_test_df.columns = ['srch_id', 'count']
srch_counts_test_df.sort_values(by='srch_id', inplace=True)
group_counts_test = srch_counts_test_df['count'].to_numpy()

query_id_test = test_df['srch_id'].to_numpy()
test_df.sort_values(by='srch_id', inplace=True)
# test_df.drop(columns=['srch_id'], inplace=True)

idx_cum_sum = np.cumsum(group_counts_test)
idx_cum_sum = idx_cum_sum -1
idx_cum_sum = np.insert(idx_cum_sum, 0, 0)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

dump_svmlight_file(X_train, y_train, 'train.txt', query_id=query_id_train)
# dump_svmlight_file(X_test, y_test, 'test.txt', query_id=query_id_test)


train_dmatrix = xgb.DMatrix('train.txt')
train_dmatrix.set_group(group_counts_train)

valid_dmatrix = xgb.DMatrix(X_val, y_val)
valid_dmatrix.set_group(group_counts_val)


# HYPERPARAMETER OPTIMIZATION IS DONE BY THIS CODE
# clf = xgb.XGBClassifier(objective = 'rank:ndcg',nthread=-1,silent =  False)

# param_grid = {        
#         'max_depth': [i for i in range(5,20,1)],
#         'learning_rate': stats.uniform(0.001,0.3),
#         'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#         'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#         'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#         'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
#         'gamma': [0, 0.25, 0.5, 1.0],
#         'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
#         'n_estimators': [100],
#         'early_stopping_rounds': [10],
#         'eval_set': [(X_val, y_val)]}


# rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=20,
#                             n_jobs=1, verbose=2, cv=2,
#                             refit=False, random_state=42)

# print("Randomized search..")
# search_time_start = time.time()
# rs_clf.fit(X_train, y_train)
# print("Randomized search time:", time.time() - search_time_start)

# best_score = rs_clf.best_score_
# best_params = rs_clf.best_params_
# print("Best score: {}".format(best_score))
# print("Best params: ")
# for param_name in sorted(best_params.keys()):
#     print('%s: %r' % (param_name, best_params[param_name]))


logger.info('Training started...')
print("Start training at {}".format(datetime.now()))
bst = xgb.train(params, train_dmatrix, num_round,  early_stopping_rounds=10,
                evals=[(valid_dmatrix, 'validation')],callbacks=callbacks)
print("Finished training at {}".format(datetime.now()))

logger.info('Training finished')

print("Start prediction at {}".format(datetime.now()))
logger.info('Start prediction...')
 
srch_query_df = test_df.copy()

# keep property ids
prop_ids = srch_query_df['prop_id'].tolist()
srch_ids = srch_query_df['srch_id'].tolist()

# drop srch_id and prop_id
srch_query_df.drop(columns=['srch_id','prop_id'], inplace=True)

# turn data into numpy array
X_test = srch_query_df.to_numpy()

# use the same scaler from before to standardize data
X_test = scaler.transform(X_test)

# create DMatrix for xgboost
test_dmatrix = xgb.DMatrix(X_test)

# do predictions for this srch_query
pred = bst.predict(test_dmatrix)

# create df for this prediction and append to final result_df
single_result_df = pd.DataFrame({"srch_id": srch_ids, "prop_id": prop_ids, "pred": pred})
single_result_df.sort_values(by='pred', ascending=False, inplace=True)

# add data to corresponding list
srch_id_list = single_result_df['srch_id'].tolist()
prop_id_list = single_result_df['prop_id'].tolist()

print("Finished prediction at {}".format(datetime.now()))
logger.info('Finished prediction...')

prediction_result_df = pd.DataFrame({'srch_id': srch_id_list, 'prop_id': prop_id_list})
print(prediction_result_df.head(20))

prediction_result_df.to_csv('predictionsVU87.csv', index=False)
logger.info('Prediction File saved!')
    
# srch_id_list = []
# prop_id_list = []

# def current_and_next(some_iterable):
#     start, stop = tee(some_iterable, 2)
#     stop = chain(islice(stop, 1, None), [None])
#     return zip(start, stop)

# print("Start prediction at {}".format(datetime.now()))
# logger.info('Start prediction...')

# for item, nxt in tqdm(current_and_next(idx_cum_sum)):
#     # need to increase start by one for all except first one, so index does not 
#     # get duplicated
#     if item != 0:
#         start = item + 1
#     else:
#         start = 0

#     stop = nxt
    
#     if stop is not None:
#         srch_query_df = test_df.iloc[start:stop,:]

#         # keep property ids
#         prop_ids = srch_query_df['prop_id'].tolist()
#         srch_ids = srch_query_df['srch_id'].tolist()

#         # drop srch_id and prop_id
#         srch_query_df.drop(columns=['srch_id','prop_id'], inplace=True)

#         # turn data into numpy array
#         X_test = srch_query_df.to_numpy()

#         # use the same scaler from before to standardize data
#         X_test = scaler.transform(X_test)

#         # create DMatrix for xgboost
#         test_dmatrix = xgb.DMatrix(X_test)

#         # do predictions for this srch_query
#         pred = bst.predict(test_dmatrix)
        
#         # create df for this prediction and append to final result_df
#         single_result_df = pd.DataFrame({"srch_id": srch_ids, "prop_id": prop_ids, "pred": pred})
#         single_result_df.sort_values(by='pred', ascending=False, inplace=True)

#         # add data to corresponding list
#         srch_id_list.extend(single_result_df['srch_id'].tolist())
#         prop_id_list.extend(single_result_df['prop_id'].tolist())
#         #prediction_result_df = prediction_result_df.append(single_result_df)

# print("Finished prediction at {}".format(datetime.now()))
# logger.info('Finished prediction...')

# prediction_result_df = pd.DataFrame({'srch_id': srch_id_list, 'prop_id': prop_id_list})
# print(prediction_result_df.head(20))

# prediction_result_df.to_csv('predictionsVU87.csv', index=False)
# logger.info('Prediction File saved!')

    
    
    
    
    
    
    