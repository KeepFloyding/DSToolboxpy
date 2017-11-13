# Automated statistic report

In a large dataset, it can sometimes be hard to determine how certain variables influence each other. Here we present a procedure that uses guidelines to get a better understanding of the data.

## Procedure

* Clean the data: the data can have missing or incomplete values and this can have an impact on the correlation of the data. 
* Outliers: there are certain outliers that are so far removed from the data and are so few that they are probably not worth consider. Despite this, they can still influence results.
* Correlation: check what kind of correlation there is between values and what their p-values are.
* Linear fit: Afterwards, we check if there is a linear fit, what is the slope and what is its p-value
* Binning: Split into distinct categories and determine if there are key differences between them using the p-value.


## Script

```
# ----------------------------------------------------------------------
# Preliminary
# ----------------------------------------------------------------------

# Load the data
df = pd.read_csv('csv_file.csv')

# Clean and filter the data
df = filter_data(df)

# ----------------------------------------------------------------------
# Determining correlation and finalising the clean
# ----------------------------------------------------------------------

# Determine the correlation of parameters with different methods
df.corr(method='pearson') # Assuming linear fit
df.corr(method='spearman') # Monotonic function
df.corr(method='kendall') # Different procedure

# Plot the key desired feature against the best correlated parameters
df_score = checkLinearFit(df,x_array,y_array,ncols, nrows)

# Determine any outliers and refined df again
df = df[new_constraint]

# Replot and checking linear fit
df_score = checkLinearFit(df,x_array,y_array,ncols, nrows)

# ----------------------------------------------------------------------
# Binning and determining key differences
# ----------------------------------------------------------------------

# Bin into seperate categories
bins = [-0.1, 1, 100]
group_names = ['Inactive','Active']

df['categories'] = pd.cut(df['play_video_tch'],bins, labels=group_names)

# Plotting histograms and determining means
df[df['categories']=='Inactive'][feature].hist(bins=100)
df[df['categories']=='Active'][feature].hist(alpha=0.5,bins=100)

print('Mean of inactive:',df[df['categories']=='Inactive'][feature].mean())
print('Mean of active:',df[df['categories']=='Active'][feature].mean())

# Doing some independance tests
ttest_ind(df[df['categories']=='Inactive'][feature],df[df['categories']=='Active'][feature],equal_var=False)

```