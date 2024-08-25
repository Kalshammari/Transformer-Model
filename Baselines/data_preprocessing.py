# -*- coding: utf-8 -*-


#In the following notebook, we are performing the statistical summarization on the original SWAN-SF dataset.

#We are calculating 5 stats - Median, Standard Deviation, Skewness, Kurtosis, and the Last Value of every feature at every instance.

#The shape of our original was - (69189, 24, 60)
#After the statistical summarization - (69189, 120) # 24*5 = 120

# Imports


import pandas as pd
import numpy as np
import scipy
from scipy.stats import skew
from scipy.stats import kurtosis

"""# Data loading

Importing the original data
"""

p1_data = pd.read_pickle(r'../partition1_data.pkl')
p2_data = pd.read_pickle(r'../partition2_data.pkl')
p3_data = pd.read_pickle(r'../partition3_data.pkl')
p4_data = pd.read_pickle(r'../partition4_data.pkl')
p5_data = pd.read_pickle(r'../partition5_data.pkl')
p1_labels = pd.read_pickle(r'../partition1_labels.pkl')
p2_labels = pd.read_pickle(r'../partition2_labels.pkl')
p3_labels = pd.read_pickle(r'../partition3_labels.pkl')
p4_labels = pd.read_pickle(r'../partition4_labels.pkl')
p5_labels = pd.read_pickle(r'../partition5_labels.pkl')

"""# Imp Functions

Setting the column name for our old data and new data
"""

og_columns = ['TOTUSJH','TOTBSQ','TOTPOT','TOTUSJZ','ABSNJZH','SAVNCPP','USFLUX','TOTFZ','MEANPOT','EPSZ',
              'MEANSHR','SHRGT45','MEANGAM','MEANGBT','MEANGBZ','MEANGBH','MEANJZH','TOTFY','MEANJZD',
              'MEANALP','TOTFX','EPSY','EPSX','R_VALUE']
new_columns = ['TOTUSJH_med','TOTUSJH_sd','TOTUSJH_sk','TOTUSJH_kt','TOTUSJH_lv',
              'TOTBSQ_med','TOTBSQ_sd','TOTBSQ_sk','TOTBSQ_kt','TOTBSQ_lv',
              'TOTPOT_med','TOTPOT_sd','TOTPOT_sk','TOTPOT_kt','TOTPOT_lv',
              'TOTUSJZ_med','TOTUSJZ_sd','TOTUSJZ_sk','TOTUSJZ_kt','TOTUSJZ_lv',
              'ABSNJZH_med','ABSNJZH_sd','ABSNJZH_sk','ABSNJZH_kt','ABSNJZH_lv',
              'SAVNCPP_med','SAVNCPP_sd','SAVNCPP_sk','SAVNCPP_kt','SAVNCPP_lv',
              'USFLUX_med','USFLUX_sd','USFLUX_sk','USFLUX_kt','USFLUX_lv',
              'TOTFZ_med','TOTFZ_sd','TOTFZ_sk','TOTFZ_kt','TOTFZ_lv',
              'MEANPOT_med','MEANPOT_sd','MEANPOT_sk','MEANPOT_kt','MEANPOT_lv',
              'EPSZ_med','EPSZ_sd','EPSZ_sk','EPSZ_kt','EPSZ_lv',
              'MEANSHR_med','MEANSHR_sd','MEANSHR_sk','MEANSHR_kt','MEANSHR_lv',
              'SHRGT45_med','SHRGT45_sd','SHRGT45_sk','SHRGT45_kt','SHRGT45_lv',
              'MEANGAM_med','MEANGAM_sd','MEANGAM_sk','MEANGAM_kt','MEANGAM_lv',
              'MEANGBT_med','MEANGBT_sd','MEANGBT_sk','MEANGBT_kt','MEANGBT_lv',
              'MEANGBZ_med','MEANGBZ_sd','MEANGBZ_sk','MEANGBZ_kt','MEANGBZ_lv',
              'MEANGBH_med','MEANGBH_sd','MEANGBH_sk','MEANGBH_kt','MEANGBH_lv',
              'MEANJZH_med','MEANJZH_sd','MEANJZH_sk','MEANJZH_kt','MEANJZH_lv',
              'TOTFY_med','TOTFY_sd','TOTFY_sk','TOTFY_kt','TOTFY_lv',
              'MEANJZD_med','MEANJZD_sd','MEANJZD_sk','MEANJZD_kt','MEANJZD_lv',
              'MEANALP_med','MEANALP_sd','MEANALP_sk','MEANALP_kt','MEANALP_lv',
              'TOTFX_med','TOTFX_sd','TOTFX_sk','TOTFX_kt','TOTFX_lv',
              'EPSY_med','EPSY_sd','EPSY_sk','EPSY_kt','EPSY_lv',
              'EPSX_med','EPSX_sd','EPSX_sk','EPSX_kt','EPSX_lv',
              'R_VALUE_med','R_VALUE_sd','R_VALUE_sk','R_VALUE_kt','R_VALUE_lv']

"""The calculate_descriptive_features function will take the dataframe as an input and will return the same datatype.

The function will take one instance at a time and caluclate the descriptive features of each column.
"""

def calculate_descriptive_features(data:DataFrame)-> DataFrame: #Finished!
    variates_to_calc_on = og_columns
    features_to_return = new_columns

    # Create empty data frame for return with named columns
    df = pd.DataFrame(columns=features_to_return)


    # For each element append to temp list
    list2add = []
    for d in variates_to_calc_on:
        l = data[d].to_numpy()
        median = np.median(l)
        last_value = data[d].iat[-1]
        std = np.std(l)
        sk = skew(l)
        kt = kurtosis(l)
        list2add.append(median)
        list2add.append(std)
        list2add.append(sk)
        list2add.append(kt)
        list2add.append(last_value)
        continue

    df.loc[len(df)] = list2add
    return list2add

"""The feature_extract function will then iterate over the original data and give one instance at a time input to the above described function.

Finally, the feature_extract function will append the data to new dataframe.
"""

def feature_extract(data):

    data_new = pd.DataFrame(columns = new_columns)

    for i in data:
        df = pd.DataFrame(i)
        temp = df.T
        temp.columns = og_columns
        to_append = calculate_descriptive_features(temp)
        df_length = len(data_new)
        data_new.loc[df_length] = to_append

    return data_new

"""# Feature Extraction

### (Median, Standard Deviation, Skewness, Kurtosis)

Calling the function for conversion
"""

p1_data_new = feature_extract(p1_data)
p2_data_new = feature_extract(p2_data)
p3_data_new = feature_extract(p3_data)
p4_data_new = feature_extract(p4_data)
p5_data_new = feature_extract(p5_data)

"""The resultant data"""

p1_data_new

"""Saving the data to csv file"""

p1_data_new.to_csv('p1_data_new.csv')
p2_data_new.to_csv('p2_data_new.csv')
p3_data_new.to_csv('p3_data_new.csv')
p4_data_new.to_csv('p4_data_new.csv')
p5_data_new.to_csv('p5_data_new.csv')

"""Reference - https://github.com/Mroussell/swan_sf"""
