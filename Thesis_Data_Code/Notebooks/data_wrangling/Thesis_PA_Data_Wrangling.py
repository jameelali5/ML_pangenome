# %%
# Data Wrangling Imports
import pandas as pd
import numpy as np
from functools import reduce

# %%
# assign to url variable for each csv file 

metadata_path = '/Users/jameelali/Desktop/Machine_Learning_Project/Khaledi Data/metadata/phenotypes.txt'
gene_presence_path = '/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/PA_gene_presence_absence.csv'
pop_struc_path = '/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/PA_population_structure.csv'
runinfo_path = '/Users/jameelali/Desktop/Machine_Learning_Project/Khaledi Data/metadata/runinfo.csv'

# load csv files into the notebook
metadata = pd.read_csv(metadata_path, sep = '\t')
gene_presence_data = pd.read_csv(gene_presence_path)
pop_struc_data = pd.read_csv(pop_struc_path)
runinfo_data = pd.read_csv(runinfo_path)

# %%
# Visualize the metadata
metadata
# 414 samples in the dataset, 5 drugs

# %%
# Visualize runinfo df
runinfo_data

# minizmize to column 'Run' and 'SampleName'
runinfo_name_col = runinfo_data[['Run', 'SampleName']]
runinfo_name_col.rename(columns = {'Run':'Isolate', 'SampleName': 'final_all'}, inplace=True)
runinfo_name_col

# %%
# Merge metadata and runinfo_name_col
metadata = pd.merge(runinfo_name_col, metadata, on='final_all')
metadata

# %%
# Drop 'final_all' and rename all columns
metadata.drop(columns=['final_all'], inplace=True)
metadata.rename(columns={ 
        'Tobramycin_S-vs-R' : 'TBM',
        'Ceftazidim_S-vs-R' : 'CTZ',
        'Ciprofloxacin_S-vs-R' : 'CIP',
        'Meropenem_S-vs-R' : 'MEM',
        'Colistin_S-vs-R' : 'COL'}, inplace=True)

# %%
# Replace NaN with 1, Replace 1.0 with R, Replace 0.0 with S
metadata.fillna(1.0, inplace=True)
metadata.replace({1.0 : 'R', 0.0 : 'S'}, regex=True, inplace=True)

# %%
# Visualize the changes
metadata
# Dataset only contains 'Isolate' and Drugs

# %%
# Save as CSV
metadata.to_csv('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/PA_metadata_final.csv', index=False)

#####################################################

# %%
# Visualize gene_presence_absence data
gene_presence_data

# %%
# Drop columns from dataset
PA_gpa = gene_presence_data.drop(columns=['Non-unique Gene name', 
'Annotation', 'No. isolates', 'No. sequences', 'Avg sequences per isolate', 
'Genome Fragment', 'Order within Fragment', 'Accessory Fragment', 
'Accessory Order with Fragment', 'QC', 'Min group size nuc', 'Max group size nuc', 
'Avg group size nuc'])
# rename the 'Gene' to 'Isolate'
PA_gpa.rename(columns={'Gene' : 'Isolate'}, inplace=True)

# %%
# one-hot encode the gpa df
PA_gpa_one_hot = PA_gpa.notnull()*1
PA_gpa_one_hot

# %%
# Replace the 'Isolate' column 
PA_gpa_one_hot['Isolate'] = PA_gpa[['Isolate']]


# %%
# Transpose the df
PA_gpa_final = PA_gpa_one_hot.T


# %%
# Visualize the df
PA_gpa_final

# %%
# Save to a CSV
PA_gpa_final.to_csv('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/PA_gpa_final.csv', header=False)

#######################################################

# %%
# Visualize the dataset
pop_struc_data

# %%
# Rename the 'Unnamed' column
pop_struc_data.rename(columns={'Unnamed: 0' : 'Isolate'}, inplace=True)

# %%
# Replace 'Cl' and 'copy' with a space
pop_struc_data.replace('Cl', '', regex=True, inplace=True)
pop_struc_data.replace('copy', '', regex=True, inplace=True)

# %%
# Visualize the dataframe
pop_struc_data

# %%
# Save to a CSV
pop_struc_data.to_csv('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/PA_ps_final.csv', index = False)

######################################################### Merge all dfs into 1 master df

# %%
metadata_final_path = '/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/PA_metadata_final.csv'
PA_gpa_final_path = '/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/PA_gpa_final.csv'
pop_struc_data_final_path = '/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/PA_ps_final.csv'

# load csv files into the notebook
metadata = pd.read_csv(metadata_final_path)
gpa = pd.read_csv(PA_gpa_final_path)
pop_struc = pd.read_csv(pop_struc_data_final_path)

# %%
metadata
# %%
gpa
# %%
pop_struc

# %%
# List of all 3 data sources
df_list = [metadata,gpa,pop_struc]

# Creating a single dataframe with all drugs and features available
Drug_df = reduce(lambda  left,right: pd.merge(left,right,on=['Isolate'], how='inner'), df_list)
Drug_df
# 400 ecoli isolates, 31242 columns (Isolate, Drugs, Genes, Clusters, Years)

# %%
# Exports the dataframe into a CSV file
Drug_df.to_csv('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/PA_Merged_dfs.csv', index= False)
# %%

