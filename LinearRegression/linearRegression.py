# These are the packages that we are importing that shall be used throughout this Lab
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

lin_reg_df = pd.read_csv('realestate.csv')
lin_reg_df.head()

lin_reg_df.isnull().sum()

new_column_names = {
    'No': 'SL No',
    'X1 transaction date': 'Txn_Dt',
    'X2 house age': 'H_Age',
    'X3 distance to the nearest MRT station': 'Distance',
    'X4 number of convenience stores': 'Conv_stores',
    'X5 latitude': 'Lat',
    'X6 longitude': 'Long',
    'Y house price of unit area': 'Price_Area'
}
lin_reg_df = lin_reg_df.rename(columns=new_column_names)

X = lin_reg_df[['H_Age', 'Distance', 'Conv_stores']]
y = lin_reg_df['Price_Area']

random_state_list = [0, 50, 101]
min_MAE, min_MSE, min_RMSE, best_rdm_st = float('inf'), float('inf'), float('inf'), 0

for rdm_st in random_state_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rdm_st)
    model_LR = LinearRegression()
    model_LR.fit(X_train, y_train)
    y_pred = model_LR.predict(X_test)

    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)

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

best_st = best_rdm_st
print(best_st)  # -- 1.1
best_MAE = min_MAE
print(best_MAE)  # -- 1.2
best_MSE = min_MSE
print(best_MSE)  # -- 1.3
best_RMSE = min_RMSE
print(best_RMSE)  # -- 1.4

coefficients = model_LR.coef_
most_sig_wt = max(coefficients)
most_sig_col = X.columns[
    np.argmax(coefficients)]  # Put the most significant column name here -- 1.5 of Gradescope tests
print(most_sig_col)  # -- 1.5

intercept_val = round(model_LR.intercept_, 2)  # Put the value here -- 1.6 of Gradescope tests
print(intercept_val)  # -- 1.6
