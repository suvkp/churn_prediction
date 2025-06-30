import numpy as np
import pandas as pd
from data_pipeline.transform.churn_label import GenerateChurnLabels

train_2008 = pd.read_parquet('data/train_2008.parquet')

# print(train_2008.head())
# print(train_2008.info())

def is_same_column(year_start=2008, year_end=2023):
    num_col_map = {}
    for y in range(year_start, year_end+1):
        temp = pd.read_parquet(f'data/train_{y}.parquet')
        num_col_map[f'train_{y}'] = temp.shape
    return num_col_map

# col_check_dict = is_same_column()
# print(col_check_dict)
# print(sum([v[0] for v in col_check_dict.values()])) # total number of rows

def pick_a_customerid_with_min_n_rows(df, n=100):
    """
    Picks a customer ID that has at least n rows in the DataFrame.
    If no such customer exists, returns None.
    """
    customer_counts = df['customer_id'].value_counts()
    valid_customers = customer_counts[customer_counts >= n].index
    if valid_customers.empty:
        return None
    return valid_customers[0] # Pick the first valid customer ID

customer = pick_a_customerid_with_min_n_rows(train_2008, n=10)
print(f"Customer ID with at least 100 rows: {customer}")

def get_first_null_value_index(df):
    """
    Returns the index of the first row in the DataFrame that contains a null value.
    If no null values are found, returns None.
    """
    for i, row in df.iterrows():
        if row.isnull().any():
            return i
    return None

train_2008_with_label = GenerateChurnLabels(inactivity_threshold=90).transform(train_2008)
print(train_2008_with_label.head())
print(train_2008_with_label.info())
customer_173 = train_2008_with_label[train_2008_with_label['customer_id'] == customer].reset_index(drop=True)
print(customer_173[['customer_id', 'date', 'activity_flag', 'churn_label']])
first_null_index = get_first_null_value_index(customer_173)
print("First null value index in customer 173 data:", first_null_index)
print(customer_173[['customer_id', 'date', 'activity_flag', 'churn_label']].iloc[first_null_index])
print(customer_173[['customer_id', 'date', 'activity_flag', 'churn_label']].iloc[first_null_index-10:first_null_index+10])

#        customer_id       date  activity_flag  churn_label
# 11526          345 2008-03-25              1          0.0
# 11527          345 2008-03-26              1          0.0
# 11528          345 2008-03-27              1          0.0
# 11529          345 2008-03-28              1          0.0
# 11530          345 2008-03-29              1          0.0
# ...            ...        ...            ...          ...
# 11724          345 2008-11-02              1          NaN
# 11725          345 2008-11-03              1          NaN
# 11726          345 2008-11-04              1          NaN
# 11727          345 2008-11-05              1          NaN
# 11728          345 2008-11-06              1          NaN

# last step was generating churn labels. There are more than one record being labeled as None for each customer. Figure out why.
# The churn label is None if the last activity date is within 90 days of the last date in the dataset.