# %%
# import libraries
from tkinter.messagebox import YES
from turtle import color
import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_inline
import re


# %%
# load the dataframe csv
path = '/Users/jameelali/Desktop/Machine Learning Project/Ecoli_Project'
drugs_feat_imp_final = path + '/drugs_feat_imp_final.csv'


# %%
# read in the csv
drugs_feat_imp_final_df = pd.read_csv(drugs_feat_imp_final, index_col=False)

# %%
# check the dataframe
drugs_feat_imp_final_df

# there should be 18270 rows (features) and 601 drugs (columns)

# %%
# check if an missing values
drugs_feat_imp_final_df.isna().sum()

# %%
# rename the first column -> Features
drugs_feat_imp_final_df = drugs_feat_imp_final_df.rename(columns={'Unnamed: 0' : 'Features'})

# %%
# check to see if index column is named 'Features'
drugs_feat_imp_final_df.head()

# %%
# filter df for only the the feature names
feature_names = drugs_feat_imp_final_df[['Features']]

feature_names

# %%
# filter df based on the string of the header
CTZ_df = drugs_feat_imp_final_df.filter(regex='^CTZ')
CTX_df = drugs_feat_imp_final_df.filter(regex='^CTX')
AMP_df = drugs_feat_imp_final_df.filter(regex='^AMP')
AMX_df = drugs_feat_imp_final_df.filter(regex='^AMX')
AMC_df = drugs_feat_imp_final_df.filter(regex='^AMC')
TZP_df = drugs_feat_imp_final_df.filter(regex='^TZP')
CXM_df = drugs_feat_imp_final_df.filter(regex='^CXM')
CET_df = drugs_feat_imp_final_df.filter(regex='^CET')
GEN_df = drugs_feat_imp_final_df.filter(regex='^GEN')
TBM_df = drugs_feat_imp_final_df.filter(regex='^TBM')
TMP_df = drugs_feat_imp_final_df.filter(regex='^TMP')
CIP_df = drugs_feat_imp_final_df.filter(regex='^CIP')


# %%
# create and average column for each of the drug frames
CTZ_df['Mean_Importance_CTZ'] = CTZ_df.mean(axis = 1)
CTX_df['Mean_Importance_CTX'] = CTX_df.mean(axis = 1)
AMP_df['Mean_Importance_AMP'] = AMP_df.mean(axis = 1)
AMX_df['Mean_Importance_AMX'] = AMX_df.mean(axis = 1)
AMC_df['Mean_Importance_AMC'] = AMC_df.mean(axis = 1)
TZP_df['Mean_Importance_TZP'] = TZP_df.mean(axis = 1)
CXM_df['Mean_Importance_CXM'] = CXM_df.mean(axis = 1)
CET_df['Mean_Importance_CET'] = CET_df.mean(axis = 1)
GEN_df['Mean_Importance_GEN'] = GEN_df.mean(axis = 1)
TBM_df['Mean_Importance_TBM'] = TBM_df.mean(axis = 1)
TMP_df['Mean_Importance_TMP'] = TMP_df.mean(axis = 1)
CIP_df['Mean_Importance_CIP'] = CIP_df.mean(axis = 1)

# %%
# merge each drug df with feature names
CTZ_feat_imp = feature_names.join(CTZ_df)
CTX_feat_imp = feature_names.join(CTX_df)
AMP_feat_imp = feature_names.join(AMP_df)
AMX_feat_imp = feature_names.join(AMX_df)
AMC_feat_imp = feature_names.join(AMC_df)
TZP_feat_imp = feature_names.join(TZP_df)
CXM_feat_imp = feature_names.join(CXM_df)
CET_feat_imp = feature_names.join(CET_df)
GEN_feat_imp = feature_names.join(GEN_df)
TBM_feat_imp = feature_names.join(TBM_df)
TMP_feat_imp = feature_names.join(TMP_df)
CIP_feat_imp = feature_names.join(CIP_df)

# %%
# filter dataframes for only feature names and mean importance
# each dataframe should have 18720 rows and 2 columns

CTZ_feat_imp = CTZ_feat_imp[['Features', 'Mean_Importance_CTZ']]
CTX_feat_imp = CTX_feat_imp[['Features', 'Mean_Importance_CTX']] 
AMP_feat_imp = AMP_feat_imp[['Features', 'Mean_Importance_AMP']]
AMX_feat_imp = AMX_feat_imp[['Features', 'Mean_Importance_AMX']]
AMC_feat_imp = AMC_feat_imp[['Features', 'Mean_Importance_AMC']]
TZP_feat_imp = TZP_feat_imp[['Features', 'Mean_Importance_TZP']]
CXM_feat_imp = CXM_feat_imp[['Features', 'Mean_Importance_CXM']]
CET_feat_imp = CET_feat_imp[['Features', 'Mean_Importance_CET']]
GEN_feat_imp = GEN_feat_imp[['Features', 'Mean_Importance_GEN']]
TBM_feat_imp = TBM_feat_imp[['Features', 'Mean_Importance_TBM']]
TMP_feat_imp = TMP_feat_imp[['Features', 'Mean_Importance_TMP']]
CIP_feat_imp = CIP_feat_imp[['Features', 'Mean_Importance_CIP']]

# %%
# sort all values from largest to smallest an take only the first 10
sorted_CTZ_imp_df = CTZ_feat_imp.sort_values(by=['Mean_Importance_CTZ'], ascending=False).head(10)
sorted_CTX_imp_df = CTX_feat_imp.sort_values(by=['Mean_Importance_CTX'], ascending=False).head(10)
sorted_AMP_imp_df = AMP_feat_imp.sort_values(by=['Mean_Importance_AMP'], ascending=False).head(10)
sorted_AMX_imp_df = AMX_feat_imp.sort_values(by=['Mean_Importance_AMX'], ascending=False).head(10)
sorted_AMC_imp_df = AMC_feat_imp.sort_values(by=['Mean_Importance_AMC'], ascending=False).head(10)
sorted_TZP_imp_df = TZP_feat_imp.sort_values(by=['Mean_Importance_TZP'], ascending=False).head(10)
sorted_CXM_imp_df = CXM_feat_imp.sort_values(by=['Mean_Importance_CXM'], ascending=False).head(10)
sorted_CET_imp_df = CET_feat_imp.sort_values(by=['Mean_Importance_CET'], ascending=False).head(10)
sorted_GEN_imp_df = GEN_feat_imp.sort_values(by=['Mean_Importance_GEN'], ascending=False).head(10)
sorted_TBM_imp_df = TBM_feat_imp.sort_values(by=['Mean_Importance_TBM'], ascending=False).head(10)
sorted_TMP_imp_df = TMP_feat_imp.sort_values(by=['Mean_Importance_TMP'], ascending=False).head(10)
sorted_CIP_imp_df = CIP_feat_imp.sort_values(by=['Mean_Importance_CIP'], ascending=False).head(10)



# %%
# subplotting all  the dfs

# mapping subplot variables
rows = 2
cols = 2
c = ['silver', 'firebrick', 'burlywood', 'lavender', 'olive', 'salmon', 'bisque', 'royalblue', 'chocolate', 'gold', 'forestgreen', 'gainsboro']
c_CTZ = ['#07f49e', '#07f49e', '#07f49e', '#07f49e', '#07f49e', '#42047e', '#42047e', '#07f49e', '#07f49e', '#07f49e']
c_CTX = ['#07f49e', '#07f49e', '#07f49e', '#07f49e', '#42047e', '#42047e', '#07f49e', '#07f49e', '#42047e', '#42047e']
c_TBM = ['#07f49e', '#07f49e', '#42047e', '#07f49e', '#07f49e', '#07f49e', '#07f49e', '#07f49e', '#07f49e', '#42047e']
c_CIP = ['#07f49e', '#07f49e', '#07f49e', '#07f49e', '#07f49e', '#07f49e', '#07f49e', '#07f49e', '#07f49e', '#07f49e']
c_GEN = ['#07f49e', '#07f49e', '#07f49e', '#07f49e', '#42047e', '#42047e', '#07f49e', '#07f49e', '#07f49e', '#07f49e']
ec = 'black'


# mapping x and y variables
features = [sorted_CTZ_imp_df['Features'],
            sorted_CTX_imp_df['Features'],
            sorted_AMP_imp_df['Features'],
            sorted_AMX_imp_df['Features'],
            sorted_AMC_imp_df['Features'],
            sorted_TZP_imp_df['Features'],
            sorted_CXM_imp_df['Features'],
            sorted_CET_imp_df['Features'],
            sorted_GEN_imp_df['Features'],
            sorted_TBM_imp_df['Features'],
            sorted_TMP_imp_df['Features'],
            sorted_CIP_imp_df['Features']]

importance = [sorted_CTZ_imp_df['Mean_Importance_CTZ'],
            sorted_CTX_imp_df['Mean_Importance_CTX'],
            sorted_AMP_imp_df['Mean_Importance_AMP'],
            sorted_AMX_imp_df['Mean_Importance_AMX'],
            sorted_AMC_imp_df['Mean_Importance_AMC'],
            sorted_TZP_imp_df['Mean_Importance_TZP'],
            sorted_CXM_imp_df['Mean_Importance_CXM'],
            sorted_CET_imp_df['Mean_Importance_CET'],
            sorted_GEN_imp_df['Mean_Importance_GEN'],
            sorted_TBM_imp_df['Mean_Importance_TBM'],
            sorted_TMP_imp_df['Mean_Importance_TMP'],
            sorted_CIP_imp_df['Mean_Importance_CIP']]

# figure size
fig, subs = plt.subplots(rows, cols, figsize=(30,25), sharex= True)
fig.suptitle('Gradient Boosted Tree Feature Importance', fontsize = 90)
fig.subplots_adjust(hspace = 0.33)


# check the 2d array designation for the subplots
'''for row in range(rows):
    for col in range(cols):
        subs[row, col].text(0.5, 0.5, 
                          str((row, col)),
                          color="green",
                          fontsize=18, 
                          ha='center')'''


# subplot for CTZ
subs[0, 0].barh(features[0], importance[0], color = c_CTZ, edgecolor = ec, linewidth = 2)
subs[0, 0].set_title('Ceftazidime (CTZ)', fontsize = '45')
for s in ['top', 'right']:
    subs[0, 0].spines[s].set_visible(False)
subs[0, 0].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[0, 0].yaxis.set_tick_params(pad = 10, labelsize = 20)
subs[0, 0].invert_yaxis()

# subplot for CTX
subs[0, 1].barh(features[1], importance[1], color = c_CTX, edgecolor = ec, linewidth = 2)
subs[0, 1].set_title('Cefotaxime (CTX)', fontsize = '45')
for s in ['top', 'right']:
    subs[0, 1].spines[s].set_visible(False)
subs[0, 1].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[0, 1].yaxis.set_tick_params(pad = 10, labelsize = 20)
subs[0, 1].invert_yaxis()

# subplot for TBM
subs[1, 0].barh(features[9], importance[9], color = c_TBM, edgecolor = ec, linewidth = 2)
subs[1, 0].set_title('Tobramycin (TBM)', fontsize = '45')
for s in ['top', 'right']:
    subs[1, 0].spines[s].set_visible(False)
subs[1, 0].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[1, 0].yaxis.set_tick_params(pad = 10, labelsize = 20)
subs[1, 0].invert_yaxis()

# subplot for CIP
subs[1, 1].barh(features[11], importance[11], color = c_CIP, edgecolor = ec, linewidth = 2)
subs[1, 1].set_title('Ciprofloxin (CIP)', fontsize = '45')
for s in ['top', 'right']:
    subs[1, 1].spines[s].set_visible(False)
subs[1, 1].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 20)
subs[1, 1].yaxis.set_tick_params(pad = 10, labelsize = 20)
subs[1, 1].invert_yaxis()

# %%
fig.savefig('feat_imp_plot_draft_2.jpg')



# %%
# subplot for GEN
fig, subs = plt.subplots( figsize=(25,25), sharex= True)
fig.suptitle('GBT Feature Importance - GEN', fontsize = 90)
fig.subplots_adjust(hspace = 0.33)

subs.barh(features[8], importance[8], color = c_GEN, edgecolor = ec, linewidth = 2)
for s in ['top', 'right']:
    subs.spines[s].set_visible(False)
subs.xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 35)
subs.yaxis.set_tick_params(pad = 10, labelsize = 35)
subs.invert_yaxis()
# %%
