from DSToolbox import *
import pandas as pd

# ---------------------------------------
# Read the sample data
# ---------------------------------------
df = pd.read_csv('sample-data.csv')

# ---------------------------------------
# Group dataframe into event legs
# ---------------------------------------

dt = 60*60*24*30 # Each event leg represents 30 days
start_date = 'registration_date'
date_feature = 'date_of_event'
group_array = ['user_id','event_leg','event_type']

df_grp = prepare_df(df,start_date, dt, date_feature, group_array)

# Save the csv for inspection
df_grp.csv('sample-grouped.csv')

# ---------------------------------------
# Assign event legs 0,1,2 as seperate features
# ---------------------------------------

month_array = [0,1,2]
type_operation = 'append' # other option is to sum
df_new = prepare_time_series(df_grp, month_array, type_operation)

# Save the csv for inspection
df_new.csv('sample-appended.csv')

