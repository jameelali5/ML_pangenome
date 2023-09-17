# %% 
# Data Wrangling Imports
import pandas as pd
import numpy as np
from functools import reduce
import csv

# %%
# Upload the gene_presence_absence_400_ecoli.csv
ecoli_gpa = pd.read_csv('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/gene_presence_absence_400_ecoli.csv')

# %%
# Check the data
ecoli_gpa.head()

# %%
# Drop columns from dataset
ecoli_400_gpa = ecoli_gpa.drop(columns=['Non-unique Gene name', 
'Annotation', 'No. isolates', 'No. sequences', 'Avg sequences per isolate', 
'Genome Fragment', 'Order within Fragment', 'Accessory Fragment', 
'Accessory Order with Fragment', 'QC', 'Min group size nuc', 'Max group size nuc', 
'Avg group size nuc'])
ecoli_400_gpa = ecoli_400_gpa.rename(columns={'Gene' : 'Isolate'})

# %%
# Check the dataframe
ecoli_400_gpa

# %%
# convert to boolean values then convert to intgers
ecoli_400_gpa_one_hot = ecoli_400_gpa.notnull()*1
ecoli_400_gpa_one_hot

# %%
# Replace the 'Isolate' column
ecoli_400_gpa_one_hot['Isolate'] = ecoli_400_gpa[['Isolate']]

# %%
# Check the dataframe
ecoli_400_gpa_one_hot

# %%
# Transpose the dataframe
ecoli_400_gpa_final = ecoli_400_gpa_one_hot.T
ecoli_400_gpa_final

# %%
# Save dataframe to csv
ecoli_400_gpa_final.to_csv('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/ecoli_400_gpa_final.csv', header=False)

################################################################

# %%
# Upload the population_structure_400_ecoli.csv
ecoli_ps = pd.read_csv('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/population_structure_400_ecoli.csv')

# %%
# Rename the unnamed column to 'Isolate'
ecoli_ps = ecoli_ps.rename(columns={'Unnamed: 0' : 'Isolate'})
ecoli_ps

# %%
# Check the shape of the dataframe, 400 samples, 298 cutoffs producing different clusters
ecoli_ps.shape

# %%
ecoli_ps = ecoli_ps.replace('Cl', '', regex=True)


# %%
# Save dataframe to csv
ecoli_ps.to_csv('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/ecoli_400_ps_final.csv', index = False)

##################################################################

# %%
# Upload Moradigaravand Metadata and runinfo
moradigaravand_metadata = pd.read_csv('/Users/jameelali/Desktop/Machine_Learning_Project/Moradigaravand Data/Raw Data Reads/MG_Accession_IDs.csv')

# %%
# Check the dataframe
moradigaravand_metadata

# %%
# Drop unneeded columns
moradigaravand_metadata = moradigaravand_metadata.drop(columns=['ENA.Accession.Number', 'Isolate', 'Sequecning Status'])

# %%
# Parse the first 400 used in the analysis and rename first colum to 'Isolate'
ecoli_400_metadata = moradigaravand_metadata.head(400).rename(columns={'Lane.accession' : 'Isolate'})

# %%
# Replace 'I' (intermediate) with 'R' (resistant)
ecoli_400_metadata = ecoli_400_metadata.replace('I', 'R', regex=True)


# %%
# Save dataframe to csv
ecoli_400_metadata.to_csv('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/ecoli_400_metadata.csv', index = False)
# %%
