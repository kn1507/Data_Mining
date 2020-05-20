
## I am grouping the count of hotels per month - For example if search date was march(Month-3) then, count of hotels booked in March 3 
will be there. 
##Full_df -> training set
##test_df -> test set
##Splitting the date in training
full_df['year'] =pd.DatetimeIndex(full_df['date_time']).year
full_df['month']=pd.DatetimeIndex(full_df['date_time']).month
##Splitting the date in training
test_df['year'] =pd.DatetimeIndex(test_df['date_time']).year
test_df['month']=pd.DatetimeIndex(test_df['date_time']).month

##creating a dataframe for groupby operation in test and train
most_searched_hotel = pd.DataFrame(full_df.groupby(['prop_id','month'])['booking_bool'].count().reset_index())
most_searched_hotel.columns = ['prop_id', 'month','monthly_booking']
most_searched_hotel.sort_values(by='monthly_booking', ascending=False)

most_searched_hotel_test = pd.DataFrame(test_df.groupby(['prop_id','month'])['booking_bool'].count().reset_index())
most_searched_hotel_test.columns = ['prop_id', 'month','monthly_booking']
most_searched_hotel_test.sort_values(by='monthly_booking', ascending=False)

##mergin the individual df with the train and test set
test_df = pd.merge(test_df, most_searched_hotel_test, how='left', left_on=['prop_id','month'], 
                     right_on=['prop_id','month'])
full_df = pd.merge(full_df, most_searched_hotel, how='left', left_on=['prop_id','month'], 
                     right_on=['prop_id','month'])
