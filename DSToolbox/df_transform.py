# Package for dataframe transformation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Function to prepare timeseries dataframe. Divides timeseries data into event_legs and groups by group_array
Inputs:
    - df: dataframe with timeseries dataframe
    - start_date: pandas series with time from which to subtract the time of event_legs
    - dt: how many seconds to consider per interval
    - date_feature: name of the time feature in the dataframe
    - group_array: how to group the dataframe

Outputs:
    - df_grp: dataframe where timeseries has been grouped into event_legs
"""

def prepare_df(df,start_date, dt, date_feature, group_array):

    df[date_feature] = pd.to_datetime(df[date_feature])
    df[start_date] = pd.to_datetime(df[start_date])
    df['event_leg'] = np.floor((df[date_feature] - df[start_date]).dt.total_seconds()/dt)

    #df.drop('date_created',axis=1)
    df_grp = df.groupby(group_array).count();
    df_grp = df_grp.xs(date_feature, axis=1, drop_level=True)

    df_grp = df_grp.unstack().fillna(0)
    #df_grp = df_grp.reset_index()

    return df_grp

"""
A function that prepares a time series dataframe in one of 2 manners:
    - sum (for every month of desired data, simply sum the values)
    - append (for every month of desired data, add columns showing activity during that month)
    
Returns a churn dataframe that corresponds to when each user churned
"""

def prepare_time_series(df, month_array, type_operation, groupby_feature = 'user_id'):
    
    # If we want to append each month as a seperate feature 
    if type_operation == 'append':
        
        # Find the months of interest
        df_new = df[df.index.get_level_values(1).isin(month_array)].unstack().fillna(0)
        
        # Name new columns
        new_cols = [str(item[0]) + '_' + str(int(item[1])) for item in df_new.columns]
        
        # Drop level and rename
        df_new.columns = df_new.columns.droplevel(0)
        df_new.columns = new_cols
    
    # If we want to sum the values of each feature for every month
    elif type_operation == 'sum':
        df_new = df[df.index.get_level_values(1).isin(month_array)]
        df_new = df_new.reset_index().groupby(groupby_feature).sum()
    
    churn = df.reset_index().groupby(groupby_feature)['event_leg'].max()
    churn = churn.rename('last_month')
    df_new = df_new.join(churn)

    return df_new

"""
Function that transforms the timeseries dataframe into a dataframe of features for each series. 
Inputs are: 
    - df: dataframe to transform_dataframe
    - month_array: array of event legs to consider as features for each user
    - df_type: Either sum the features of each event_leg or attach them as seperate features
Outputs:
    - df_trf: transformed dataframe
    - label: array of labels of users who have churned or not in the following event leg (1 for churn, 0 for no churn)
"""
    
def transform_dataframe(df, month_array=[0], df_type='append',**kwargs):
    
    # To predict user dropoff, we need to choose how many months of user data to incorporate. 
    # In other words, given X months of user data, will the user still be on the platform in month X+1
    
    last_month = max(month_array) # Last month of activity
    
    # The function below adds months of user data as seperate features with a corresponding churn array
    
    df_trf = prepare_time_series_2(df, month_array, df_type,**kwargs)
    
    # Eliminate all the users who have churned before month X
    df_trf = df_trf[df_trf['last_month']>=last_month] # users must have last month equal or greater than month X
    churn = df_trf['last_month']
    
    # Label the remaining users. Those who churned at month X+1 (last month is X), establish as 1, otherwise 0
    label = [1 if item == last_month else 0 for item in churn.values]
    
    return df_trf, label