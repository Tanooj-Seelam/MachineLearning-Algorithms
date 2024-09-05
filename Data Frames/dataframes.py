import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

df = pd.read_csv('realestate.csv')
df.head()

shape_data = df.shape  # 1.1 of Gradescope tests
print(shape_data)

df.info()

df.describe()
X2_max = df['X2 house age'].max()  # Put the value of the max value of X2 house age here -- 1.2 of Gradescope tests
X4_max = df['X4 number of convenience stores'].max()  # Put the value of the max value of X4 number of convenience stores here -- 1.3 of Gradescope tests

df.isnull().sum()
X3_null = df['X3 distance to the nearest MRT station'].isnull().sum()  # Put the value here -- 1.4 of Gradescope tests
X4_null = df['X4 number of convenience stores'].isnull().sum()  # Put the value here -- 1.5 of Gradescope tests
Y_null = df['Y house price of unit area'].isnull().sum()  # Put the value here -- 1.6 of Gradescope tests

df_drop = df.copy()
df_drop = df_drop.dropna(axis=0)
mean_X3_drop = df_drop['X3 distance to the nearest MRT station'].mean()  # Put the mean value calculated here -- 1.7 of Gradescope tests
print(mean_X3_drop)

df_fill = df.copy()
df_fill = df_fill.fillna(df_fill.median())
mean_X3_fill = df_fill['X3 distance to the nearest MRT station'].mean()  # Put the mean value calculated here -- 1.8 of Gradescope tests
print(mean_X3_fill)

dataframe = df_fill.copy()
dataframe = dataframe[dataframe['Y house price of unit area'] <= 80]
dataframe = dataframe[dataframe['X3 distance to the nearest MRT station'] <= 2800]
dataframe = dataframe[dataframe['X6 longitude'] >= 121.50]

conversion_factor = 91
dataframe['Y house price of unit area'] = dataframe['Y house price of unit area'] * conversion_factor

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalize = dataframe.copy()
df_normalize = scaler.fit_transform(df_normalize)

normalized_df = pd.DataFrame(df_normalize,columns=dataframe.columns)
mean_norm_Y = normalized_df['Y house price of unit area'].mean()  # Put the mean value of the 'Y house price of unit area' column after Norm, here - 1.9 of Gradescope tests

def age_classifier(age):
    if age < 10:
        return 'New'
    elif age < 30:
        return 'Middle'
    else:
        return 'Old'


df_age_classify = dataframe.copy()
df_age_classify['Age Class'] = df_age_classify['X2 house age'].apply(age_classifier)

New_count = (df_age_classify['Age Class'] == 'New').sum()  # Count of houses classified as 'New' -- 1.10 of Gradescope tests
Middle_count = (df_age_classify['Age Class'] == 'Middle').sum()  # Count of houses classified as 'Middle' -- 1.11 of Gradescope tests
Old_count = (df_age_classify['Age Class'] == 'Old').sum()  # Count of houses classified as 'Old' -- 1.12 of Gradescope tests

reindex_df = dataframe.set_index('No')

max_id = reindex_df['Y house price of unit area'].idxmax()  # Put the value here -- 1.13 of Gradescope tests

txn_dt = reindex_df.loc[max_id, 'X1 transaction date']  # Put the transaction date value here -- 1.14 of Gradescope tests
house_age = reindex_df.loc[max_id, 'X2 house age']  # Put the House Age value here -- 1.15 of Gradescope tests
conv_st = reindex_df.loc[max_id, 'X4 number of convenience stores']  # Put the Number of Convenience stores value here -- 1.16 of Gradescope tests

age_price_df = dataframe[(dataframe['X2 house age'] <= 9) & (dataframe['Y house price of unit area'] > 27)]

grouped_age_price_df = age_price_df.groupby('X4 number of convenience stores')['Y house price of unit area'].mean()

mean_val_conv_7 = grouped_age_price_df.get(7, 0)  # Put the value here -- 1.17 of Gradescope tests

grouped_age_price_df.plot(kind='bar', title='Mean House Price based on Convenience store proximity',
                          ylabel='Price of unit area', xlabel='Number of convenience stores', figsize=(6, 5))
plt.show()
