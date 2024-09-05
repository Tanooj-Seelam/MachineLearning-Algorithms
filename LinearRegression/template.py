# These are the packages that we are importing that shall be used throughout this Lab

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Read the dataset and look at the head to gague the column names and whether column names actually exists

lin_reg_df = #YOUR CODE HERE

lin_reg_df.head()

# Perform the basic Null Check to decide whether imputation or drop is required

lin_reg_df.isnull().sum()

# Now rename the column names to make it easy for your own usage - use the rename() function
# Use new column names as: 'SL No', 'Txn_Dt', 'H_Age', 'Distance', 'Conv_stores', 'Lat', 'Long', 'Price_Area'

lin_reg_df.rename() #YOUR CODE HERE

# Split the dataset into target and feature values such that you consider only the following features: House Age, Distance to MRT station and Number of Convenience stores
# While we consider Price per Unit Area as the Traget variable

y = #YOUR CODE HERE

X = #YOUR CODE HERE

# After that test the model with random_state - 0, 50 and 101 and report the one that gave the best performance based on MSE, MAE and RMSE

random_state_list = [0, 50, 101]

min_MAE, min_MSE, min_RMSE, best_rdm_st = float('inf'), float('inf'), float('inf'), 0

for rdm_st in random_state_list:

    X_train, X_test, y_train, y_test = #YOUR CODE HERE - train-test split: 75 - 25

    model_LR = #YOUR CODE HERE - init the Linear Regression model

    #YOUR CODE HERE - fit the data into the model

    #YOUR CODE HERE - Predict using this model

    # Use sklearn.metrics to get the values of MAE and MSE

    MAE = #YOUR CODE HERE
    MSE = #YOUR CODE HERE
    RMSE = #YOUR CODE HERE -- remember RMSE is sqare root of MSE
    
    if MAE < min_MAE:
        
        min_MAE = MAE
        min_MSE = MSE
        min_RMSE = RMSE
        best_rdm_st = rdm_st
        

    print("For random state = {}, the values are: ".format(rdm_st))
    print("Mean Absolute Error: ", MAE)
    print("Mean Squared Error: ", MSE)
    print("Root Mean Squared Error: ", RMSE)
    print("========================================================")
    print("\n")

# Report the the random state that gave the best result and the respective values of MAE, MSE and RMSE

best_st = # Put the value of best random state here -- 1.1 of Gradescope tests
print(best_st) # -- 1.1

best_MAE = # Put the value of best MAE here -- 1.2 of Gradescope tests
print(best_MAE) # -- 1.2

best_MSE = # Put the value of best MSE here -- 1.3 of Gradescope tests
print(best_MSE) # -- 1.3

best_RMSE = # Put the value of best RMSE here -- 1.4 of Gradescope tests
print(best_RMSE) # -- 1.4

# based on the value of these coefficients, indicate which column(s) seem to be the most significant contributor to the LR model == Convenience Stores (as per our model)

most_sig_wt, idx = 0, 0

for index, wt in enumerate(): #YOUR CODE HERE inside the brackets -- use the ceoff_ param of LinearRegression()
    
    if most_sig_wt < abs(wt):
        
        most_sig_wt = wt
        idx = index

most_sig_col = # Put the most significant column name here -- 1.5 of Gradescope tests

print(most_sig_col) # -- 1.5

# what is the intercept, for the best model as chosen by you
# Use intercept_ param of LinearRegression() and round it to 2 decimal places

intercept_val = # Put the value here -- 1.6 of Gradescope tests

print(intercept_val) # -- 1.6