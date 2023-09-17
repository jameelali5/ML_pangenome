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
Feat_Imp_df = pd.read_csv(path+"GB_feat_imp_df_400_ecoli.csv", na_values="NaN")
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
# Check the dataframe
Feat_Imp_df

# %%
# create empty dictionaries to hold drug values
CTZ_df = {}
CXM_df = {}
TZP_df = {}
CTX_df = {}
CET_df = {}
GEN_df = {}
AMX_df = {}
AMC_df = {}
TBM_df = {}
TMP_df = {}
AMP_df = {}
CIP_df = {}

# %%
# loop to split data into smaller dataframes
for drug in drug_list:
    data = Feat_Imp_df.loc[Feat_Imp_df["Drug_Combo"].str.startswith(drug+'_GYS')]
    # print (data)
    if drug == drug_list[0]:
        CTZ_df = data
    if drug == drug_list[1]:
        CXM_df = data
    if drug == drug_list[2]:
        TZP_df = data
    if drug == drug_list[3]:
        CTX_df = data
    if drug == drug_list[4]:
        CET_df = data
    if drug == drug_list[5]:
        GEN_df = data
    if drug == drug_list[6]:
        AMX_df = data
    if drug == drug_list[7]:
        AMC_df = data
    if drug == drug_list[8]:
        TBM_df = data
    if drug == drug_list[9]:
        TMP_df = data
    if drug == drug_list[10]:
        AMP_df = data
    if drug == drug_list[11]:
        CIP_df = data

# %%
# Check one of the df
CTZ_df 

# %%
# Create a new column with all possible drug combinations
CTZ_df[['Drug', 'Combo', 'Iteration']] = CTZ_df['Drug_Combo'].str.split('_', expand= True)
CXM_df[['Drug', 'Combo', 'Iteration']] = CXM_df['Drug_Combo'].str.split('_', expand= True)
TZP_df[['Drug', 'Combo', 'Iteration']] = TZP_df['Drug_Combo'].str.split('_', expand= True)
CTX_df[['Drug', 'Combo', 'Iteration']] = CTX_df['Drug_Combo'].str.split('_', expand= True)
CET_df[['Drug', 'Combo', 'Iteration']] = CET_df['Drug_Combo'].str.split('_', expand= True)
GEN_df[['Drug', 'Combo', 'Iteration']] = GEN_df['Drug_Combo'].str.split('_', expand= True)
AMX_df[['Drug', 'Combo', 'Iteration']] = AMX_df['Drug_Combo'].str.split('_', expand= True)
AMC_df[['Drug', 'Combo', 'Iteration']] = AMC_df['Drug_Combo'].str.split('_', expand= True)
TBM_df[['Drug', 'Combo', 'Iteration']] = TBM_df['Drug_Combo'].str.split('_', expand= True)
TMP_df[['Drug', 'Combo', 'Iteration']] = TMP_df['Drug_Combo'].str.split('_', expand= True)
AMP_df[['Drug', 'Combo', 'Iteration']] = AMP_df['Drug_Combo'].str.split('_', expand= True)
CIP_df[['Drug', 'Combo', 'Iteration']] = CIP_df['Drug_Combo'].str.split('_', expand= True)

# %%
# Drop 'Drug_Combo' col and replace 'NaN' with 0
CTZ_df = CTZ_df.drop(columns=['Drug_Combo']).fillna(0)
CXM_df = CXM_df.drop(columns=['Drug_Combo']).fillna(0)
TZP_df = TZP_df.drop(columns=['Drug_Combo']).fillna(0)
CTX_df = CTX_df.drop(columns=['Drug_Combo']).fillna(0)
CET_df = CET_df.drop(columns=['Drug_Combo']).fillna(0)
GEN_df = GEN_df.drop(columns=['Drug_Combo']).fillna(0)
AMX_df = AMX_df.drop(columns=['Drug_Combo']).fillna(0)
AMC_df = AMC_df.drop(columns=['Drug_Combo']).fillna(0)
TBM_df = TBM_df.drop(columns=['Drug_Combo']).fillna(0)
TMP_df = TMP_df.drop(columns=['Drug_Combo']).fillna(0)
AMP_df = AMP_df.drop(columns=['Drug_Combo']).fillna(0)
CIP_df = CIP_df.drop(columns=['Drug_Combo']).fillna(0)

# %%
# check data frame
CTZ_df

# %%
# Take the mean of each row and minimize the dfs
CTZ_mean_df = CTZ_df.mean(axis=0).to_frame(name='CTZ_Importance').drop(index='Iteration')
CXM_mean_df = CXM_df.mean(axis=0).to_frame(name='CXM_Importance').drop(index='Iteration')
TZP_mean_df = TZP_df.mean(axis=0).to_frame(name='TZP_Importance').drop(index='Iteration')
CTX_mean_df = CTX_df.mean(axis=0).to_frame(name='CTX_Importance').drop(index='Iteration')
CET_mean_df = CET_df.mean(axis=0).to_frame(name='CET_Importance').drop(index='Iteration')
GEN_mean_df = GEN_df.mean(axis=0).to_frame(name='GEN_Importance').drop(index='Iteration')
AMX_mean_df = AMX_df.mean(axis=0).to_frame(name='AMX_Importance').drop(index='Iteration')
AMC_mean_df = AMC_df.mean(axis=0).to_frame(name='AMC_Importance').drop(index='Iteration')
TBM_mean_df = TBM_df.mean(axis=0).to_frame(name='TBM_Importance').drop(index='Iteration')
TMP_mean_df = TMP_df.mean(axis=0).to_frame(name='TMP_Importance').drop(index='Iteration')
AMP_mean_df = AMP_df.mean(axis=0).to_frame(name='AMP_Importance').drop(index='Iteration')
CIP_mean_df = CIP_df.mean(axis=0).to_frame(name='CIP_Importance').drop(index='Iteration')

# %%
# Check the dfs
CIP_mean_df

# %%
# Drop rows that == 0
CTZ_mean_df.drop(CTZ_mean_df[CTZ_mean_df['CTZ_Importance'] == 0].index, inplace=True)
CXM_mean_df.drop(CXM_mean_df[CXM_mean_df['CXM_Importance'] == 0].index, inplace=True)
TZP_mean_df.drop(TZP_mean_df[TZP_mean_df['TZP_Importance'] == 0].index, inplace=True)
CTX_mean_df.drop(CTX_mean_df[CTX_mean_df['CTX_Importance'] == 0].index, inplace=True)
CET_mean_df.drop(CET_mean_df[CET_mean_df['CET_Importance'] == 0].index, inplace=True)
GEN_mean_df.drop(GEN_mean_df[GEN_mean_df['GEN_Importance'] == 0].index, inplace=True)
AMX_mean_df.drop(AMX_mean_df[AMX_mean_df['AMX_Importance'] == 0].index, inplace=True)
AMC_mean_df.drop(AMC_mean_df[AMC_mean_df['AMC_Importance'] == 0].index, inplace=True)
TBM_mean_df.drop(TBM_mean_df[TBM_mean_df['TBM_Importance'] == 0].index, inplace=True)
TMP_mean_df.drop(TMP_mean_df[TMP_mean_df['TMP_Importance'] == 0].index, inplace=True)
AMP_mean_df.drop(AMP_mean_df[AMP_mean_df['AMP_Importance'] == 0].index, inplace=True)
CIP_mean_df.drop(CIP_mean_df[CIP_mean_df['CIP_Importance'] == 0].index, inplace=True)

# %%
# Check the dfs
CIP_mean_df

# %%
# Sort all the values
CTZ_mean_df.sort_values(by=['CTZ_Importance'], ascending=False, inplace=True)
CXM_mean_df.sort_values(by=['CXM_Importance'], ascending=False, inplace=True)
TZP_mean_df.sort_values(by=['TZP_Importance'], ascending=False, inplace=True)
CTX_mean_df.sort_values(by=['CTX_Importance'], ascending=False, inplace=True)
CET_mean_df.sort_values(by=['CET_Importance'], ascending=False, inplace=True)
GEN_mean_df.sort_values(by=['GEN_Importance'], ascending=False, inplace=True)
AMX_mean_df.sort_values(by=['AMX_Importance'], ascending=False, inplace=True)
AMC_mean_df.sort_values(by=['AMC_Importance'], ascending=False, inplace=True)
TBM_mean_df.sort_values(by=['TBM_Importance'], ascending=False, inplace=True)
TMP_mean_df.sort_values(by=['TMP_Importance'], ascending=False, inplace=True)
AMP_mean_df.sort_values(by=['AMP_Importance'], ascending=False, inplace=True)
CIP_mean_df.sort_values(by=['CIP_Importance'], ascending=False, inplace=True)


# %%
# Reset the index for each df
CTZ_mean_df.reset_index(inplace=True)
CXM_mean_df.reset_index(inplace=True)
TZP_mean_df.reset_index(inplace=True)
CTX_mean_df.reset_index(inplace=True)
CET_mean_df.reset_index(inplace=True)
GEN_mean_df.reset_index(inplace=True)
AMX_mean_df.reset_index(inplace=True)
AMC_mean_df.reset_index(inplace=True)
TBM_mean_df.reset_index(inplace=True)
TMP_mean_df.reset_index(inplace=True)
AMP_mean_df.reset_index(inplace=True)
CIP_mean_df.reset_index(inplace=True)

# %%
# Rename the index column to Feature and 
CTZ_mean_df.rename(columns={'index':'Features'}, inplace=True)
CXM_mean_df.rename(columns={'index':'Features'}, inplace=True)
TZP_mean_df.rename(columns={'index':'Features'}, inplace=True)
CTX_mean_df.rename(columns={'index':'Features'}, inplace=True)
CET_mean_df.rename(columns={'index':'Features'}, inplace=True)
GEN_mean_df.rename(columns={'index':'Features'}, inplace=True)
AMX_mean_df.rename(columns={'index':'Features'}, inplace=True)
AMC_mean_df.rename(columns={'index':'Features'}, inplace=True)
TBM_mean_df.rename(columns={'index':'Features'}, inplace=True)
TMP_mean_df.rename(columns={'index':'Features'}, inplace=True)
AMP_mean_df.rename(columns={'index':'Features'}, inplace=True)
CIP_mean_df.rename(columns={'index':'Features'}, inplace=True)

# %%
# Check the dfs
CTZ_mean_df

# %%
# Export certain dfs
CIP_mean_df.to_csv(path+"ecoli_CIP_feat_imp_df.csv", index=False) # for PA
CTZ_mean_df.to_csv(path+"ecoli_CTZ_feat_imp_df.csv", index=False) # for PA
TBM_mean_df.to_csv(path+"ecoli_TBM_feat_imp_df.csv", index=False) # for PA

# %%
CTZ_head_df = CTZ_mean_df.head(10)
CXM_head_df = CXM_mean_df.head(10)
TZP_head_df = TZP_mean_df.head(10)
CTX_head_df = CTX_mean_df.head(10)
CET_head_df = CET_mean_df.head(10)
GEN_head_df = GEN_mean_df.head(10)
AMX_head_df = AMX_mean_df.head(10)
AMC_head_df = AMC_mean_df.head(10)
TBM_head_df = TBM_mean_df.head(10)
TMP_head_df = TMP_mean_df.head(10)
AMP_head_df = AMP_mean_df.head(10)
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
fig.suptitle('E.coli GB Feature Importance', fontsize = 90, weight = 'semibold')
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
subs[1, 1].barh(y = AMP_head_df['Features'], width = AMP_head_df['AMP_Importance'], edgecolor = 'gray', linewidth = 2, color = ColorPallete(AMP_head_df))
subs[1, 1].invert_yaxis()
subs[1, 1].set_title('Ampicillin (AMP)', fontsize = '45', y = 1.015)



# %%
fig.savefig('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Images/Feat_Imp_Plot_400_ecoli.jpg')# %%

# %%
