# %%
# import libraries
from tkinter.messagebox import YES
from turtle import color
import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_inline
import pandas as pd


# %%
# set path to working directory
path = '/Users/jameelali/Desktop/Machine Learning Project/Khaledi Data/'

# %%
# load PA metadata 
PA_metadata = pd.read_csv(path + 'metadata/phenotypes.txt', sep = '\t') 


# %%
# checks
PA_metadata.head()


# %%
# rename the headers isolate Isolate, TOB, CAZ, CIP, MEM, CST
PA_metadata.rename( columns= {
        'final_all' : 'Isolate', 
        'Tobramycin_S-vs-R' : 'TOB',
        'Ceftazidim_S-vs-R' : 'CAZ',
        'Ciprofloxacin_S-vs-R' : 'CIP',
        'Meropenem_S-vs-R' : 'MEM',
        'Colistin_S-vs-R' : 'CST'}, inplace=True)


# %%
# check for new header names
PA_metadata.head()


# %%
# check to see how many values are missing (intermediate resistance)
PA_metadata.isna().sum()


# %%
# fill NaN as resistant (1) and conver columns to int, susceptible is 0
PA_metadata = PA_metadata.fillna(1).astype({
        'TOB' : int,
        'CAZ' : int,
        'CIP' : int,
        'MEM' : int,
        'CST' : int})


# %%
# check counts for each column 1: Resistance, 0: Susceptible
print(PA_metadata['TOB'].value_counts()) # 0: 276, 1: 138
print(PA_metadata['CST'].value_counts()) # 0: 329, 1: 85
print(PA_metadata['CAZ'].value_counts()) # 0: 169, 1: 245
print(PA_metadata['CIP'].value_counts()) # 0: 159, 1: 255
print(PA_metadata['MEM'].value_counts()) # 0: 110, 1: 304
# authors of paper did not use CST because they decided have atleast 100 in class


# %%
# check the final dataframe, use for labels and merge on the 'Isolate' col
PA_metadata


# %%
# load PA gene_presence_absence
PA_gpa = pd.read_csv(path + 'Aim_1_Data/gene_presence_absence.csv') 


# %%
# checks gpa
PA_gpa.T

# %%
