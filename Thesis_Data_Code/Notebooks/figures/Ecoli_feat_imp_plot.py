# %%
# import libraries
from tkinter.messagebox import YES
from turtle import color
import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_inline



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

# there should be 18270 rows (features) and 601 columns

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
#test plotting

# figure size
fig, ax = plt.subplots(4, 4, figsize=(10,10))

# mapping x and y variables
features = sorted_CTZ_imp_df['Features'],
importance = sorted_CTZ_imp_df['Mean_Importance_CTZ']
c = 'olive'
ec = 'black'

# horizontal bar plot
ax.bar(features, importance, color = c, edgecolor = ec, linewidth = 2)

# remove axes spines
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# Remove x, y Ticks
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5, labelrotation = 45)
ax.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
 
# Add plot title
ax.set_title('XGBClassifier Feature Importance for CTZ', fontsize = 'xx-large', fontweight = 'bold')
 


# %%
# subplotting all  the dfs

# mapping subplot variables
rows = 4
cols = 3
c = ['silver', 'firebrick', 'burlywood', 'lavender', 'olive', 'salmon', 'bisque', 'royalblue', 'chocolate', 'gold', 'forestgreen', 'gainsboro' ]
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
fig, subs = plt.subplots(rows, cols, figsize=(50,50), sharey= True)
fig.suptitle('XGBClassifier Feature Importance per Antibiotic', fontsize = 90, fontweight = 'bold')
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
subs[0, 0].bar(features[0], importance[0], color = c[0], edgecolor = ec, linewidth = 2)
subs[0, 0].set_title('Ceftazidime (CTZ)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[0, 0].spines[s].set_visible(False)
subs[0, 0].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 25)
subs[0, 0].yaxis.set_tick_params(pad = 10)

# subplot for CTX
subs[0, 1].bar(features[1], importance[1], color = c[1], edgecolor = ec, linewidth = 2)
subs[0, 1].set_title('Cefotaxime (CTX)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[0, 1].spines[s].set_visible(False)
subs[0, 1].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 15)
subs[0, 1].yaxis.set_tick_params(pad = 10)

# subplot for AMP
subs[0, 2].bar(features[2], importance[2], color = c[2], edgecolor = ec, linewidth = 2)
subs[0, 2].set_title('Ampicillin (AMP)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[0, 2].spines[s].set_visible(False)
subs[0, 2].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 15)
subs[0, 2].yaxis.set_tick_params(pad = 10)

# subplot for AMX
subs[1, 0].bar(features[3], importance[3], color = c[3], edgecolor = ec, linewidth = 2)
subs[1, 0].set_title('Amoxicillin (AMX)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[1, 0].spines[s].set_visible(False)
subs[1, 0].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 15)
subs[1, 0].yaxis.set_tick_params(pad = 10)

# subplot for AMC
subs[1, 1].bar(features[4], importance[4], color = c[4], edgecolor = ec, linewidth = 2)
subs[1, 1].set_title('Amoxicillin - Clavulanate (AMC)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[1, 1].spines[s].set_visible(False)
subs[1, 1].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 15)
subs[1, 1].yaxis.set_tick_params(pad = 10)

# subplot for TZP
subs[1, 2].bar(features[5], importance[5], color = c[5], edgecolor = ec, linewidth = 2)
subs[1, 2].set_title('Pipericillin - Tazzobactam (TZP)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[1, 2].spines[s].set_visible(False)
subs[1, 2].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 15)
subs[1, 2].yaxis.set_tick_params(pad = 10)

# subplot for CXM
subs[2, 0].bar(features[6], importance[6], color = c[6], edgecolor = ec, linewidth = 2)
subs[2, 0].set_title('Cefuroxime (CXM)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[2, 0].spines[s].set_visible(False)
subs[2, 0].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 15)
subs[2, 0].yaxis.set_tick_params(pad = 10)

# subplot for CET
subs[2, 1].bar(features[7], importance[7], color = c[7], edgecolor = ec, linewidth = 2)
subs[2, 1].set_title('Cephalothin (CET)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[2, 1].spines[s].set_visible(False)
subs[2, 1].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 15)
subs[2, 1].yaxis.set_tick_params(pad = 10)

# subplot for GEN
subs[2, 2].bar(features[8], importance[8], color = c[8], edgecolor = ec, linewidth = 2)
subs[2, 2].set_title('Gentamicin (GEN)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[2, 2].spines[s].set_visible(False)
subs[2, 2].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 15)
subs[2, 2].yaxis.set_tick_params(pad = 10)

# subplot for TBM
subs[3, 0].bar(features[9], importance[9], color = c[9], edgecolor = ec, linewidth = 2)
subs[3, 0].set_title('Tobramycin (TBM)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[3, 0].spines[s].set_visible(False)
subs[3, 0].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 15)
subs[3, 0].yaxis.set_tick_params(pad = 10)

# subplot for TMP
subs[3, 1].bar(features[10], importance[10], color = c[10], edgecolor = ec, linewidth = 2)
subs[3, 1].set_title('Trimethoprim (TMP)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[3, 1].spines[s].set_visible(False)
subs[3, 1].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 15)
subs[3, 1].yaxis.set_tick_params(pad = 10)

# subplot for CIP
subs[3, 2].bar(features[11], importance[11], color = c[11], edgecolor = ec, linewidth = 2)
subs[3, 2].set_title('Ciprofloxin (CIP)', fontsize = '45', fontweight = 'bold')
for s in ['top', 'right']:
    subs[3, 2].spines[s].set_visible(False)
subs[3, 2].xaxis.set_tick_params(pad = 5, labelrotation = 45, labelsize = 15)
subs[3, 2].yaxis.set_tick_params(pad = 10)


# %%
fig.savefig('feat_imp_plot_draft.jpg')
# %%
