# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:02:30 2023

@author: mraja
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import csv
import math
import wrds
import datetime as dt
from linearmodels.panel import FamaMacBeth
import statsmodels.api as sm
#import psycopg2  # This import statement is commented out
import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats
from pandas.tseries.offsets import MonthEnd

# Connect to the WRDS database
db = wrds.Connection()

# Extract stock data using SQL from CRSP database for a specific time range and specific exchange codes
crsp = db.raw_sql("""
                      select a.permno, a.date, b.shrcd, b.exchcd,
                      a.ret, a.shrout, a.prc
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/1965' and '12/31/2009'
                      and b.exchcd between 1 and 3
                      """, date_cols=['date'])

# Filter the data to keep only those with specific share codes and exchange codes
crsp = crsp[(crsp['shrcd'].isin([10, 11])) & (crsp['exchcd'].between(1, 3))]

# Extract delisting returns
dlret = db.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp.msedelist
                     """, date_cols=['dlstdt'])
dlret['date'] = dlret['dlstdt'] + MonthEnd(0)

# Merge delisting return data with main stock data
crsp = pd.merge(crsp, dlret, how='left', on=['permno', 'date'])

# Extract year and month from the date column for further operations
crsp['year'] = crsp['date'].dt.year
crsp['month'] = crsp['date'].dt.month

# Calculate market equity
crsp['me'] = abs(crsp['prc']) * crsp['shrout']

# Filter out June data
june_data = crsp[crsp['month'] == 6]

# Calculate market value for June
june_data['jme'] = abs(june_data['prc']) * june_data['shrout']
june_data['year_me'] = june_data['year']

# Adjust years for non-June months
crsp['month1'] = crsp['month']
crsp['year_me'] = crsp['year'] - 1
crsp.loc[crsp['month1'] >= 7, 'year_me'] = crsp['year']

# Merge June market value data with the main data
crsp = pd.merge(crsp, june_data[['permno', 'year_me', 'jme']], on=['permno', 'year_me'], how='left')
crsp = crsp.drop(['year_me', 'month1'], axis=1)

# Calculate December market equity
decme = crsp[crsp['month'] == 12]
decme = decme.sort_values(by=['permno', 'year'])
decme['prc_shifted'] = decme.groupby('permno')['prc'].shift(1)
decme['shrout_shifted'] = decme.groupby('permno')['shrout'].shift(1)
decme['decme'] = abs(decme['prc_shifted']) * decme['shrout_shifted']
crsp = pd.merge(crsp, decme[['permno', 'year', 'decme']], on=['permno', 'year'], how='left')

# Load external CSV file containing Compustat-CRSP Merged (CCM) dataset
CCM = pd.read_csv("C:/Users/mraja/OneDrive/Desktop/Mark/New Rep/CCM.csv")

# Merge the CCM data with the main data
crsp = pd.merge(crsp, CCM[['permno', 'gvkey']], how='left', on=['permno'])
crsp = crsp.dropna(subset=['gvkey'])
crsp['year'] = crsp['date'].dt.year

# Extract financial data from the Compustat database
comp = db.raw_sql("""
                    select gvkey, datadate, at, pstkl, txditc,
                    pstkrv, seq, pstk
                    from comp.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/1965'
                    """, date_cols=['datadate'])

# Extract year from the datadate column
comp['year'] = comp['datadate'].dt.year

# Sort and group the DataFrame to fill missing values
comp.sort_values(by=['gvkey', 'year'], inplace=True)
comp = comp.groupby(['gvkey', 'year']).apply(lambda x: x.ffill().bfill())
comp = comp.drop_duplicates(subset=['gvkey', 'year'], keep='first')
comp.reset_index(drop=True, inplace=True)

# Create preferred stock and book equity measures
comp['ps'] = np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
comp['ps'] = np.where(comp['ps'].isnull(), comp['pstk'], comp['ps'])
comp['ps'] = np.where(comp['ps'].isnull(), 0, comp['ps'])
comp['txditc'] = comp['txditc'].fillna(0)
comp['be'] = comp['seq'] + comp['txditc'] - comp['ps']
comp['be'] = np.where(comp['be'] > 0, comp['be'], np.nan)
comp['be_1'] = comp.groupby('gvkey')['be'].shift(1)
comp = comp.sort_values(by=['gvkey', 'year'])

# Adjust data types for merging purposes
comp['gvkey'] = comp['gvkey'].str.lstrip('0')
comp['year'] = comp['year'].astype(str)
comp['gvkey'] = comp['gvkey'].astype(str)
crsp['year'] = crsp['year'].astype(str)
crsp['gvkey'] = crsp['gvkey'].astype(int).astype(str)

# Merge the financial data (comp) with the main data (crsp)
cc = pd.merge(crsp, comp[['be_1', 'gvkey', 'year']], how='left', on=['gvkey', 'year'])

# Filter out observations with missing book equity
cc = cc.dropna(subset=['be_1'])

# Calculate the book-to-market ratio
cc['beme'] = cc['be_1'] * 1000 / cc['decme']
cc = cc[cc['beme'] > 0]

# Create quintiles for size and B/M ratio in June
june_q = cc[cc['month'] == 6]
june_q['sizeq'] = cc.groupby('year')['jme'].transform(lambda x: pd.qcut(x, 5, labels=False))
june_q['bmq'] = cc.groupby('year')['beme'].transform(lambda x: pd.qcut(x, 5, labels=False))
cc = pd.merge(cc, june_q[['permno', 'year', 'sizeq', 'bmq']], on=['permno', 'year'], how='left')

# Shift the market equity data
cc.sort_values(by=['permno', 'year', 'month'], inplace=True)
cc['meb'] = cc.groupby('permno')['me'].shift(+1)
cc.sort_index(inplace=True)

# Convert the month and year columns back to float data types for further operations
cc['month'] = cc['month'].astype(float)
cc['year'] = cc['year'].astype(float)

# Calculate portfolio returns based on quintiles
portfolio_returns = pd.DataFrame(columns=['year', 'month', 'sizeq', 'bmq', 'style_return'])
for year in range(1965, 2010):
    for month in range(1, 13):
        monthly_data = cc[(cc['year'] == year) & (cc['month'] == month)]
        for sizeq in range(5):
            for bmq in range(5):
                selected_securities = monthly_data[
                    (monthly_data['sizeq'] == sizeq) &
                    (monthly_data['bmq'] == bmq)
                ]
                selected_securities['weight'] = selected_securities['meb'] / selected_securities['meb'].sum()
                portfolio_return = (selected_securities['ret'] * selected_securities['weight']).sum()
                portfolio_returns = portfolio_returns.append({
                    'year': year,
                    'month': month,
                    'sizeq': sizeq,
                    'bmq': bmq,
                    'style_return': portfolio_return
                }, ignore_index=True)

# Merge the portfolio returns with the main data
cc = pd.merge(cc, portfolio_returns, on=['year', 'month', 'sizeq', 'bmq'], how='left')

# Sort data for consistent processing
cc.sort_values(by=['permno', 'date'], inplace=True)

# Calculate future returns for different horizons
horizons = [1, 3, 6, 12]
for horizon in horizons:
    future_return_column = f'future_ret_{horizon}m'
    cc[future_return_column] = (
        cc.groupby('permno')['ret']
        .rolling(window=horizon)
        .apply(lambda x: np.prod(x + 1) - 1)
        .groupby('permno')
        .shift(-horizon)
        .reset_index(level=0, drop=True)
    )

# Adjust future returns for delisted companies
delisted_permnos = cc.dropna(subset=['dlstdt'])['permno'].unique()
for horizon in horizons:
    future_return_column = f'future_ret_{horizon}m'
    delisted_data = cc[cc['permno'].isin(delisted_permnos)]
    delisted_data[future_return_column] = (
        delisted_data.groupby('permno')['ret']
        .transform(lambda x: x.iloc[::-1].rolling(window=horizon, min_periods=1).apply(lambda x: np.prod(x + 1) - 1).iloc[::-1])
    )
    delisted_data[future_return_column] = delisted_data.groupby('permno')[future_return_column].shift(-1)
    cc.update(delisted_data)

# Calculate past returns for different horizons
periods = [6, 12]
for period in periods:
    past_return_column = f'past_ret_{period}m'
    cc[past_return_column] = (
        cc.groupby('permno')['ret']
        .rolling(window=period)
        .apply(lambda x: np.prod(x + 1) - 1)
        .groupby('permno')
        .shift(1)
        .reset_index(level=0, drop=True)
    )

# Calculate past portfolio returns for different horizons
for period in periods:
    style_return_column = f'style_ret_{period}m'
    cc[style_return_column] = (
        cc.groupby('permno')['style_return']
        .rolling(window=period)
        .apply(lambda x: np.prod(x + 1) - 1)
        .groupby('permno')
        .shift(1)
        .reset_index(level=0, drop=True)
    )

# Winsorize (clip extreme values) for specific columns
cc_copy = cc.copy()
columns_to_winsorize = ['jme', 'beme', 'past_ret_6m', 'past_ret_12m']
def winsorize_column(column):
    lower_percentile = 0.01
    upper_percentile = 0.99
    grouped_by_month = cc.groupby(['year', 'month'])
    for month, group_data in grouped_by_month:
        group_data[column] = mstats.winsorize(group_data[column], limits=(lower_percentile, upper_percentile))
for column in columns_to_winsorize:
    winsorize_column(column)
changes_made = not cc.equals(cc_copy)
if changes_made:
    print("Winsorization made changes to the DataFrame.")
else:
    print("No changes were made to the DataFrame.")

# Calculate logarithms for some variables
cc['log_jme'] = np.log(cc['jme'])
cc['log_beme'] = np.log(cc['beme'])

# Reset index for final dataset
cc.reset_index(inplace=True, drop=True)
cc['date'] = pd.to_datetime(cc['date'])
cc = cc.set_index(['permno', 'date'], drop=False)

# Define lists for regression variables
dependent_variables = ['future_ret_1m', 'future_ret_3m', 'future_ret_6m', 'future_ret_12m']
independent_variables = [
    ['past_ret_6m', 'log_jme', 'log_beme'],
    ['style_ret_6m', 'log_jme', 'log_beme'],
    ['style_ret_6m', 'past_ret_6m', 'log_jme', 'log_beme'],
    ['past_ret_12m', 'log_jme', 'log_beme'],
    ['style_ret_12m', 'log_jme', 'log_beme'],
    ['style_ret_12m', 'past_ret_12m', 'log_jme', 'log_beme']
]

# Run regressions and save results to a file
with open('regression_results_Final.txt', 'w') as file:
    for dep_var in dependent_variables:
        for ind_vars in independent_variables:
            Y = cc[dep_var]
            X = sm.add_constant(cc[ind_vars])
            model = FamaMacBeth(Y, X)
            res = model.fit(cov_type='kernel')
            file.write(f'Dependent Variable: {dep_var}\n')
            file.write(f'Independent Variables: {ind_vars}\n')
            file.write(f'Result:\n{res}\n\n')

# Save the final dataset to a CSV file
cc.to_csv("C:/Users/mraja/OneDrive/Desktop/cc.csv")



