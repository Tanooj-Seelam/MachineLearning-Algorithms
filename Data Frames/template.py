# These are the packages that we are importing that shall be used throughout this Lab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# %matplotlib inline --> Use this line in case you are running in a jupyter notebook

# Data set information and deep dive - https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set

# Read the dataset and look at the head to gague the column names and whether column names actually exists

df = pd.read_csv() # Enter dataset name here

df.head()

# What is the shape of the data?

shape_data = # 1.1 of Gradescope tests
print(shape_data)

# Look at the information about the data for better understanding of what you are dealing with, using - info()

df.info()

# Use the describe function to report the max values of all the rows from X2 and X4

df.describe()

X2_max = # Put the value of the max value of X2 here -- 1.2 of Gradescope tests
X4_max = # Put the value of the max value of X4 here -- 1.3 of Gradescope tests

# Use null check to report how many values are missing for the X3, X4 and Y columns

df.isnull().sum()

X3_null =  # Put the value here -- 1.4 of Gradescope tests
X4_null =  # Put the value here -- 1.5 of Gradescope tests
Y_null =  # Put the value here -- 1.6 of Gradescope tests

# Perform dropna and find the mean value of the 'X3 distance to the nearest MRT station' column

df_drop = df.copy()
df_drop = df_drop.dropna(axis = 0)

mean_X3_drop =  # Put the mean value calculated here -- 1.7 of Gradescope tests
print(mean_X3_drop)

# Perform fillna with median values and report the mean value of the 'X3 distance to the nearest MRT station' column after this - 1.7

df_fill = df.copy()

df_fill = df_fill.fillna(df_fill.median())

mean_X3_fill =  # Put the mean value calculated here -- 1.8 of Gradescope tests
print(mean_X3_fill)

# Now use the new dataframe with filled in data (which is the one you used fillna() on) for all further tasks

# Outlier removal - make sure to remove the outlier as per the following:
# remove all values in 'Y house price of unit area' column with value > 80
# remove all values in 'X3 distance to the nearest MRT station' column with value > 2800
# remove all values in 'X6 longitude' column with value < 121.50

dataframe = df_fill.copy()

dataframe = # YOUR CODE HERE

# Here the Y column has price per unit area as 10000 New Taiwan Dollar/ Ping where 1 NTD = 0.03 USD and 1 Ping = 3.3 meter^2 --> thus use the Pandas apply function to convert the unit to USD/ m^2
# Conversion facor to be used is 91  ==> current * conversion factor (91)
# To complete this, use the apply() function

dataframe['Y house price of unit area'] = # YOUR CODE HERE

# Perform Normalization on the data (hint: check MinMaxScaler()) -- report the mean of the Y column after this

from sklearn import preprocessing

normalized = # YOUR CODE HERE

# Note make sure you pay heed to the data type of normalized

normalized_df = # YOUR CODE HERE to convert normalized back to a dataframe

mean_norm_Y = # Put the mean value of the 'Y house price of unit area' column after Norm, here - 1.9 of Gradescope tests

# Use the apply function to classify whether a house should be classified as New/ Moderate/ Old
# Use this new dataset to find how many house are there for each of the 3 categories as defined above and report in the respective variables

def age_classifier(age):
    
    if age < 10:
        
        return 'New'
    
    elif age < 30:
        
        return 'Middle'
    
    else:
        
        return 'Old'


df_age_classify = dataframe.copy()

df_age_classify['Age Class'] = # YOUR CODE HERE -- perform the apply() based on the function specified above

New_count =  # Count of houses classified as 'New' -- 1.10 of Gradescope tests
Middle_count =  # Count of houses classified as 'Middle' -- 1.11 of Gradescope tests
Old_count =  # Count of houses classified as 'Old' -- 1.12 of Gradescope tests

# Reset the index and set the 'No' column as the Index (this helps to learn how to use the serial number column in dsets where it's present and can be effectively used).
# Use set_index()

reindex_df = # YOUR CODE HERE

# Find index where the column price per unit area has the maximum value

max_id =  # Put the value here -- 1.13 of Gradescope tests

# Report the transaction date, house age and number of convenience stores for this house at idmax index

txn_dt =  # Put the transaction date value here -- 1.14 of Gradescope tests
house_age =  # Put the House Age value here -- 1.15 of Gradescope tests
conv_st =  # Put the Number of Convenience stores value here -- 1.16 of Gradescope tests

# Subset the dataframe querying for House Age less than or equal to 9 years and Price of Unit Area greater than 27

age_price_df = # YOUR CODE HERE

# Group the mean of the 'Y house price of unit area' column based on the 'X4 number of convenience stores' and find out mean value of the Y col when X4 col value is 7.
# Use the groupby() function

grouped_age_price_df =  # Be wary that operation may create a pandas series object so handle accordingly

# Report the mean value when number of convenience stores = 7 --> 7th index of the series object (Use this documentation for what functions you may use to get the desired value: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html)
# round the value to the nearest 2 decimal places

mean_val_conv_7 =  # Put the value here -- 1.17 of Gradescope tests

# PLot the Bar Chart with the following specifications title = 'Mean House Price based on Convenience store proximity', ylabel= 'Price of unit area', xlabel = 'Number of convenience stores', figsize=(6, 5)
# Use plot.bar()

# YOUR CODE HERE