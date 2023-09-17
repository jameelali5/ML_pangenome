# %%
# Data Wrangling Imports
import pandas as pd
import numpy as np

# Data visualization Imports
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# %%
# Loading featuere importance dataframe
path = '/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/'
Feat_Imp_df = pd.read_csv(path+"PA_GB_feat_imp_df.csv", na_values="NaN")
Feat_Imp_df = Feat_Imp_df.rename(columns = {'Unnamed: 0':'Drug_Combo'}) # renaming the first column
Feat_Imp_df

# %%
# Extracting list of combos from dictionary created
combo_list = list(Feat_Imp_df["Drug_Combo"].str.split("_", expand= True)[1].unique())
combo_list 

# %%
# Extracting list of drugs from dictionary created
drug_list = list(Feat_Imp_df["Drug_Combo"].str[:3].unique()) 
drug_list 

# %%
# create empty dictionaries to hold drug values
TBM_df = {}
MEM_df = {}
CTZ_df = {}
COL_df = {}
CIP_df = {}

# %%
# loop to split data into smaller dataframes
for drug in drug_list:
    data = Feat_Imp_df.loc[Feat_Imp_df["Drug_Combo"].str.startswith(drug+'_GS')]
    # print (data)
    if drug == drug_list[0]:
        TBM_df = data
    if drug == drug_list[1]:
        MEM_df = data
    if drug == drug_list[2]:
        CTZ_df = data
    if drug == drug_list[3]:
        COL_df = data
    if drug == drug_list[4]:
        CIP_df = data

# %%
# Check one of the df
CTZ_df 

# %%
# Create a new column with all possible drug combinations
TBM_df[['Drug', 'Combo', 'Iteration']] = TBM_df['Drug_Combo'].str.split('_', expand= True)
MEM_df[['Drug', 'Combo', 'Iteration']] = MEM_df['Drug_Combo'].str.split('_', expand= True)
CTZ_df[['Drug', 'Combo', 'Iteration']] = CTZ_df['Drug_Combo'].str.split('_', expand= True)
COL_df[['Drug', 'Combo', 'Iteration']] = COL_df['Drug_Combo'].str.split('_', expand= True)
CIP_df[['Drug', 'Combo', 'Iteration']] = CIP_df['Drug_Combo'].str.split('_', expand= True)

# %%
# Drop 'Drug_Combo' col and replace 'NaN' with 0
TBM_df = TBM_df.drop(columns=['Drug_Combo']).fillna(0)
MEM_df = MEM_df.drop(columns=['Drug_Combo']).fillna(0)
CTZ_df = CTZ_df.drop(columns=['Drug_Combo']).fillna(0)
COL_df = COL_df.drop(columns=['Drug_Combo']).fillna(0)
CIP_df = CIP_df.drop(columns=['Drug_Combo']).fillna(0)

# %%
# check data frame
TBM_df


# %%
# Take the mean of each row and minimize the dfs
TBM_mean_df = TBM_df.mean(axis=0).to_frame(name='TBM_Importance').drop(index='Iteration')
MEM_mean_df = MEM_df.mean(axis=0).to_frame(name='MEM_Importance').drop(index='Iteration')
CTZ_mean_df = CTZ_df.mean(axis=0).to_frame(name='CTZ_Importance').drop(index='Iteration')
COL_mean_df = COL_df.mean(axis=0).to_frame(name='COL_Importance').drop(index='Iteration')
CIP_mean_df = CIP_df.mean(axis=0).to_frame(name='CIP_Importance').drop(index='Iteration')

# %%
# Check the dfs
TBM_mean_df

# %%
# Drop rows that == 0
TBM_mean_df.drop(TBM_mean_df[TBM_mean_df['TBM_Importance'] == 0].index, inplace=True)
MEM_mean_df.drop(MEM_mean_df[MEM_mean_df['MEM_Importance'] == 0].index, inplace=True)
CTZ_mean_df.drop(CTZ_mean_df[CTZ_mean_df['CTZ_Importance'] == 0].index, inplace=True)
COL_mean_df.drop(COL_mean_df[COL_mean_df['COL_Importance'] == 0].index, inplace=True)
CIP_mean_df.drop(CIP_mean_df[CIP_mean_df['CIP_Importance'] == 0].index, inplace=True)

# %%
# Check the dfs
CIP_mean_df

# %%
# Sort all the values
TBM_mean_df.sort_values(by=['TBM_Importance'], ascending=False, inplace=True)
MEM_mean_df.sort_values(by=['MEM_Importance'], ascending=False, inplace=True)
CTZ_mean_df.sort_values(by=['CTZ_Importance'], ascending=False, inplace=True)
COL_mean_df.sort_values(by=['COL_Importance'], ascending=False, inplace=True)
CIP_mean_df.sort_values(by=['CIP_Importance'], ascending=False, inplace=True)

# %%
# Reset the index for each df
TBM_mean_df.reset_index(inplace=True)
MEM_mean_df.reset_index(inplace=True)
CTZ_mean_df.reset_index(inplace=True)
COL_mean_df.reset_index(inplace=True)
CIP_mean_df.reset_index(inplace=True)

# %%
# Rename the index column to Feature, Final df
TBM_mean_df.rename(columns={'index':'Features'}, inplace=True)
MEM_mean_df.rename(columns={'index':'Features'}, inplace=True)
CTZ_mean_df.rename(columns={'index':'Features'}, inplace=True)
COL_mean_df.rename(columns={'index':'Features'}, inplace=True)
CIP_mean_df.rename(columns={'index':'Features'}, inplace=True)

# %%
# Check the dfs
CIP_mean_df

#%%
CIP_mean_df.to_csv(path+"PA_CIP_feat_imp_df.csv", index=False) # for PA
CTZ_mean_df.to_csv(path+"PA_CTZ_feat_imp_df.csv", index=False) # for PA
TBM_mean_df.to_csv(path+"PA_TBM_feat_imp_df.csv", index=False) # for PA

# %%
TBM_head_df = TBM_mean_df.head(10)
MEM_head_df = MEM_mean_df.head(10)
CTZ_head_df = CTZ_mean_df.head(10)
COL_head_df = COL_mean_df.head(10)
CIP_head_df = CIP_mean_df.head(10)



# %%
'''Function to create color pallete from dataframes'''
def ColorPallete(data_frame):
    color_list = []
    
    for feature in data_frame['Features']:
        #print (feature)
        if feature.startswith('group'):
            #print (feature)
            color_list.append('#87CBB9')
        elif feature.startswith('Cutoff'):
            #print (feature)
            color_list.append('#B9EDDD')
        else:
            #print (feature)
            color_list.append('#569DAA')

    return color_list

# %%
# Create barplots

# figure size
fig, subs = plt.subplots(nrows=2, ncols=2, figsize=(30,25), sharex= True)
fig.suptitle('P. aeruginosa GB Feature Importance', fontsize = 90, weight = 'semibold')
fig.subplots_adjust(hspace = 0.33)
for s in ['top', 'right']:
    subs[0, 0].spines[s].set_visible(False)
subs[0, 0].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[0, 0].yaxis.set_tick_params(pad = 10, labelsize = 20)

# Subplot for Tobramycin
subs[0, 0].barh(y = TBM_head_df['Features'], width = TBM_head_df['TBM_Importance'], edgecolor = 'gray', linewidth = 2, color = ColorPallete(TBM_head_df))
subs[0, 0].invert_yaxis()
subs[0, 0].set_title('Tobramycin (TBM)', fontsize = '45', y = 1.015)
for s in ['top', 'right']:
    subs[0, 1].spines[s].set_visible(False)
subs[0, 1].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[0, 1].yaxis.set_tick_params(pad = 10, labelsize = 20)

# Subplot for Ciprofloxacin
subs[0, 1].barh(y = CIP_head_df['Features'], width = CIP_head_df['CIP_Importance'], edgecolor = 'gray', linewidth = 2, color = ColorPallete(CIP_head_df))
subs[0, 1].invert_yaxis()
subs[0, 1].set_title('Ciprofloxacin (CIP)', fontsize = '45', y = 1.015)
for s in ['top', 'right']:
    subs[1, 0].spines[s].set_visible(False)
subs[1, 0].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[1, 0].yaxis.set_tick_params(pad = 10, labelsize = 20)

# Subplot for Ceftazidime
subs[1, 0].barh(y = CTZ_head_df['Features'], width = CTZ_head_df['CTZ_Importance'], edgecolor = 'gray', linewidth = 2, color = ColorPallete(CTZ_head_df))
subs[1, 0].invert_yaxis()
subs[1, 0].set_title('Ceftazidime (CTZ)', fontsize = '45', y = 1.015)
for s in ['top', 'right']:
    subs[1, 1].spines[s].set_visible(False)
subs[1, 1].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[1, 1].yaxis.set_tick_params(pad = 10, labelsize = 20)

# Subplot for Ampicillin
subs[1, 1].barh(y = COL_head_df['Features'], width = COL_head_df['COL_Importance'], edgecolor = 'gray', linewidth = 2, color = ColorPallete(MEM_head_df))
subs[1, 1].invert_yaxis()
subs[1, 1].set_title('Colistin (COL)', fontsize = '45', y = 1.015)

# %%
fig.savefig('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Images/Feat_Imp_Plot_PA.jpg', bbox_inches="tight")
# %%

# %%
