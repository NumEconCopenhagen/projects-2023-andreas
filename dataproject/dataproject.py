import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean_taxes():
    'clean the SKAT data'

    # read file and remove unnecessary rows
    taxes = pd.read_excel('SKAT.xlsx', skiprows=2)

    # remove unnecessary columns
    taxes.drop(columns = ['Unnamed: 0'], axis=1, inplace=True) # axis = 1 -> columns, inplace=True -> changed, no copy made

    # remove columns from 1947-2007 and 2022
    cols = []
    [cols.append(str(i)) for i in range(1947,2008)]
    taxes.drop(columns='2022', axis=1, inplace=True)
    taxes.drop(columns=cols, axis=1, inplace=True) # axis = 1 -> columns, inplace=True -> changed, no copy made

    # convert to billion DKK
    taxes.iloc[0,:] = taxes.iloc[0,:]/1000**2
    
    return taxes

def clean_wages():
    'clean the LOENSUM data'

    # read file and remove unnecessary rows
    wage_sum = pd.read_excel('LOENSUM.xlsx', skiprows=2)

    # remove unnecessary columns
    wage_sum.drop(columns = ['Unnamed: 1','Unnamed: 0'], axis=1, inplace=True) # axis = 1 -> columns, inplace=True -> changed, no copy made

    # convert to billion DKK
    wage_sum.iloc[0,:] = wage_sum.iloc[0,:]/1000

    return wage_sum

def plot_wages_taxes(merged):
    'plot wages and taxes from data'

    # Predefine options for plots
    plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"-"})
    plt.rcParams.update({'font.size': 10})

    # Create a subplot with three plots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(17.5, 5))

    # Plot personal income taxes
    merged.plot(x='year', y='personal_income_taxes', legend=False, ax=ax1)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Billion DKK')
    ax1.set_title('Personal Income Taxes')

    # Plot wage sum
    merged.plot(x='year', y='wage_sum', legend=False, ax=ax2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Billion DKK')
    ax2.set_title('Wage Sum')

    # Plot first differences
    merged.plot(x='year', y='wage_sum_change', ax=ax3)
    merged.plot(x='year', y='taxes_change',  ax=ax3)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Billion DKK')
    ax3.set_title('First differences')

    # Display the plot
    plt.show()