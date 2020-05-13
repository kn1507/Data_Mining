import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file
import matplotlib.pyplot as plt
from itertools import tee, islice, chain
from datetime import datetime
from tqdm import tqdm

pd.options.mode.chained_assignment = None  # default='warn'

# Begin 
print("Begin Data Processing at {}".format(datetime.now()))

# load the data
full_df = pd.read_csv("data/training_set_VU_DM.csv")

# load the test data
test_df = pd.read_csv("data/test_set_VU_DM.csv")

### Data transformations ###

# delete records where price > 25000 or 1000, only 100 examples where price booked was over 1000
full_df = full_df[full_df['price_usd']<1000]

# most booked hotels
most_booked_hotel = pd.DataFrame(full_df.groupby('prop_id')['booking_bool'].sum().reset_index())
most_booked_hotel.columns = ['prop_id', 'booking_count']
most_booked_hotel.sort_values(by='booking_count', ascending=False)

full_df = pd.merge(full_df, most_booked_hotel, how='left', left_on='prop_id', right_on='prop_id')

# occurences of hotels and click probability
occurences_prop_id = pd.DataFrame(full_df['prop_id'].value_counts().reset_index().sort_values(by='index'))
occurences_prop_id.columns = ['prop_id', 'occurence_count']
full_df = pd.merge(full_df, occurences_prop_id, how='left', left_on='prop_id', right_on='prop_id')
full_df['click_prob'] = full_df['booking_count'] / full_df['occurence_count']


### select feature columns for training and test data ###
feature_cols_train = ['srch_id', 'price_usd', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_starrating', 'booking_bool']

# there is no booking bool in test data
feature_cols_test = feature_cols_train[:-1]

# but we need prop id for sorting the ranking
feature_cols_test.append('prop_id')


# create the feature df for training
feature_df = full_df[feature_cols_train]

### dummy variables for selected columsn
feature_df = pd.get_dummies(feature_df, columns=['prop_starrating', 'prop_review_score'], prefix_sep='_', drop_first=True)

# ### normalize columns ###
# normalize_columns = ['price_usd', 'prop_location_score1']
# for c in normalize_columns:
#     data = feature_df[[c]].values.astype(float)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data_scaled = pd.DataFrame(min_max_scaler.fit_transform(data))
#     data_scaled.columns = [c]
#     feature_df[c] = data_scaled


### Random Undersampling ###

# separate booking bools and keep all the booking_bools
booking_bool_0_df = feature_df[feature_df['booking_bool'] == 0]
booking_bool_1_df = feature_df[feature_df['booking_bool'] == 1] 
unique_srch_ids_1 = booking_bool_1_df['srch_id'].unique().tolist()

### Train and Validation set ###
train_ids, validation_ids = train_test_split(unique_srch_ids_1, test_size=0.25)
book_1_train = booking_bool_1_df[booking_bool_1_df['srch_id'].isin(train_ids)]
book_1_validation = booking_bool_1_df[booking_bool_1_df['srch_id'].isin(validation_ids)]

book_0_train = booking_bool_0_df[booking_bool_0_df['srch_id'].isin(train_ids)]
book_0_validation = booking_bool_0_df[booking_bool_0_df['srch_id'].isin(validation_ids)]

train_df = pd.concat([book_0_train, book_1_train])
validation_df = pd.concat([book_0_validation, book_1_validation])

### deal with nan values ###
### replace them here to not leak information from training to testing data
### replace nan with mean for prop_review_score and prop_location_score_2
train_df = train_df.fillna(train_df.mean())
validation_df = validation_df.fillna(validation_df.mean())


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

# get needed arrays for svmlight_dump
query_id_train = train_df['srch_id'].to_numpy()
y_train = train_df['booking_bool'].to_numpy()
train_df.drop(columns=['srch_id','booking_bool'], inplace=True)
X_train = train_df.to_numpy()

query_id_val = validation_df['srch_id'].to_numpy()
y_val = validation_df['booking_bool'].to_numpy()
validation_df.drop(columns=['srch_id', 'booking_bool'], inplace=True)
X_val = validation_df.to_numpy()

# get test data in correct format
test_df = test_df[feature_cols_test]
test_df.sort_values(by='srch_id', inplace=True)

# get corresponding srch_id counts
srch_counts_test_df = pd.DataFrame(test_df['srch_id'].value_counts().reset_index())
srch_counts_test_df.columns = ['srch_id', 'count']
srch_counts_test_df.sort_values(by='srch_id', inplace=True)
group_counts_test = srch_counts_test_df['count'].to_numpy()

# get list of indices so test data can be predicted by respective query group
idx_cum_sum = np.cumsum(group_counts_test)
idx_cum_sum = idx_cum_sum -1
idx_cum_sum = np.insert(idx_cum_sum, 0, 0)

# create the same dummy variables for test data
test_df = pd.get_dummies(test_df, columns=['prop_starrating', 'prop_review_score'], prefix_sep='_', drop_first=True)
# test__ids = test_df['srch_id'].to_numpy()
# test_df.drop(columns=['srch_id'], inplace=True)
# X_test = test_df.to_numpy()

# normalize X datasets
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print("End Data Processing at {}".format(datetime.now()))

### dump pandas df to svmlight_file to train.txt and group input ad train.txt.group
dump_svmlight_file(X_train, y_train, 'train.txt', query_id=query_id_train)

#np.savetxt("train.txt.group", group_counts_train, fmt="%i")

### XGBOOST ###
train_dmatrix = xgb.DMatrix('train.txt')
train_dmatrix.set_group(group_counts_train)

valid_dmatrix = xgb.DMatrix(X_val, y_val)
valid_dmatrix.set_group(group_counts_val)


params = {'objective': 'rank:pairwise'}

num_round = 10

print("Start training at {}".format(datetime.now()))
bst = xgb.train(params, train_dmatrix, evals=[(valid_dmatrix, 'validation')])
print("Finished training at {}".format(datetime.now()))

# # save the model
# bst.save_model('lambda1.model')

# xgb.plot_importance(bst)
# plt.show()

# predictions have to be done one group at a time

srch_id_list = []
prop_id_list = []

def current_and_next(some_iterable):
    start, stop = tee(some_iterable, 2)
    stop = chain(islice(stop, 1, None), [None])
    return zip(start, stop)

print("Start prediction at {}".format(datetime.now()))

for item, nxt in tqdm(current_and_next(idx_cum_sum)):
    # need to increase start by one for all except first one, so index does not 
    # get duplicated
    if item != 0:
        start = item + 1
    else:
        start = 0

    stop = nxt
    
    if stop is not None:
        srch_query_df = test_df.iloc[start:stop,:]

        # keep property ids
        prop_ids = srch_query_df['prop_id'].tolist()
        srch_ids = srch_query_df['srch_id'].tolist()

        # drop srch_id and prop_id
        srch_query_df.drop(columns=['srch_id', 'prop_id'], inplace=True)

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
        srch_id_list.extend(single_result_df['srch_id'].tolist())
        prop_id_list.extend(single_result_df['prop_id'].tolist())
        #prediction_result_df = prediction_result_df.append(single_result_df)

print("Finished prediction at {}".format(datetime.now()))

prediction_result_df = pd.DataFrame({'srch_id': srch_id_list, 'prop_id': prop_id_list})
print(prediction_result_df.head(20))

prediction_result_df.to_csv('predictionsVU87.csv', index=False)
