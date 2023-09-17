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
ecoli_CIP_df = pd.read_csv(path+"ecoli_CIP_feat_imp_df.csv", na_values="NaN")
ecoli_CTZ_df = pd.read_csv(path+"ecoli_CTZ_feat_imp_df.csv", na_values="NaN")
ecoli_TBM_df = pd.read_csv(path+"ecoli_TBM_feat_imp_df.csv", na_values="NaN")

PA_CIP_df = pd.read_csv(path+"PA_CIP_feat_imp_df.csv", na_values="NaN")
PA_CTZ_df = pd.read_csv(path+"PA_CTZ_feat_imp_df.csv", na_values="NaN")
PA_TBM_df = pd.read_csv(path+"PA_TBM_feat_imp_df.csv", na_values="NaN")

# %%
# Check each df
ecoli_TBM_df

# %%
# get the first 10 features from each dataframe
ecoli_CIP_df_head = ecoli_CIP_df.head(10)
ecoli_CTZ_df_head = ecoli_CTZ_df.head(10)
ecoli_TBM_df_head = ecoli_TBM_df.head(10)
PA_CIP_df_head = PA_CIP_df.head(10)
PA_CTZ_df_head = PA_CTZ_df.head(10)
PA_TBM_df_head = PA_TBM_df.head(10)

# %%
# check each df
PA_CIP_df_head

# %%
'''Function to create color pallete from dataframes'''
def ColorPallete(data_frame):
    color_list = []
    
    for feature in data_frame['Features']:
        #print (feature)
        if feature.startswith('group'):
            #print (feature)
            color_list.append('#83C5BE')
        elif feature.startswith('Cutoff'):
            #print (feature)
            color_list.append('#EDF6F9')
        else:
            #print (feature)
            color_list.append('#006D77')

    return color_list

# %%
# Create Barplots

# figure size
fig, subs = plt.subplots(nrows=3, ncols=2, figsize=(40,35), sharex= 'row')
fig.suptitle('E. coli vs. P. aeruginosa Feature Importance', fontsize = 90, weight = 'semibold')
fig.subplots_adjust(hspace = 0.33, wspace= 0.33)


# Subplot for ecoli vs Ciprofloxacin
subs[0, 0].barh(y = ecoli_CIP_df_head['Features'], width = ecoli_CIP_df_head['CIP_Importance'], edgecolor = 'gray', linewidth = 2, color = ColorPallete(ecoli_CIP_df_head))
subs[0, 0].invert_yaxis()
subs[0, 0].set_title('E. coli vs. CIP', fontsize = '55', y = 1.015)
for s in ['top', 'right']:
    subs[0, 0].spines[s].set_visible(False)
subs[0, 0].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[0, 0].yaxis.set_tick_params(pad = 10, labelsize = 30)

# Subplot for PA vs Ciprofloxacin
subs[0, 1].barh(y = PA_CIP_df_head['Features'], width = PA_CIP_df_head['CIP_Importance'], edgecolor = 'gray', linewidth = 2, color = ColorPallete(PA_CIP_df_head))
subs[0, 1].invert_yaxis()
subs[0, 1].set_title('P. aeruginosa vs. CIP', fontsize = '55', y = 1.015)
for s in ['top', 'right']:
    subs[0, 1].spines[s].set_visible(False)
subs[0, 1].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[0, 1].yaxis.set_tick_params(pad = 10, labelsize = 30)

# Subplot for ecoli vs Ceftazidime
subs[1, 0].barh(y = ecoli_CTZ_df_head['Features'], width = ecoli_CTZ_df_head['CTZ_Importance'], edgecolor = 'gray', linewidth = 2, color = ColorPallete(ecoli_CTZ_df_head))
subs[1, 0].invert_yaxis()
subs[1, 0].set_title('E. coli vs. CTZ', fontsize = '55', y = 1.015)
for s in ['top', 'right']:
    subs[1, 0].spines[s].set_visible(False)
subs[1, 0].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[1, 0].yaxis.set_tick_params(pad = 10, labelsize = 30)

# Subplot for PA vs Ceftazidime
subs[1, 1].barh(y = PA_CTZ_df_head['Features'], width = PA_CTZ_df_head['CTZ_Importance'], edgecolor = 'gray', linewidth = 2, color = ColorPallete(PA_CTZ_df_head))
subs[1, 1].invert_yaxis()
subs[1, 1].set_title('P. aeruginosa vs CTZ', fontsize = '55', y = 1.015)
for s in ['top', 'right']:
    subs[1, 1].spines[s].set_visible(False)
subs[1, 1].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[1, 1].yaxis.set_tick_params(pad = 10, labelsize = 30)

# Subplot for ecoli vs Tobramycin
subs[2, 0].barh(y = ecoli_TBM_df_head['Features'], width = ecoli_TBM_df_head['TBM_Importance'], edgecolor = 'gray', linewidth = 2, color = ColorPallete(ecoli_TBM_df_head))
subs[2, 0].invert_yaxis()
subs[2, 0].set_title('E. coli vs. TBM', fontsize = '55', y = 1.015)
for s in ['top', 'right']:
    subs[2, 0].spines[s].set_visible(False)
subs[2, 0].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[2, 0].yaxis.set_tick_params(pad = 10, labelsize = 30)

# Subplot for PA vs Tobramycin
subs[2, 1].barh(y = PA_TBM_df_head['Features'], width = PA_TBM_df_head['TBM_Importance'], edgecolor = 'gray', linewidth = 2, color = ColorPallete(PA_TBM_df_head))
subs[2, 1].invert_yaxis()
subs[2, 1].set_title('P. aeruginosa vs. TBM', fontsize = '55', y = 1.015)
for s in ['top', 'right']:
    subs[2, 1].spines[s].set_visible(False)
subs[2, 1].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[2, 1].yaxis.set_tick_params(pad = 10, labelsize = 30)


# %%
fig.savefig('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Images/Combiened_Feat_Imp_Plot_PA.jpg', bbox_inches="tight")# %%

# %%
