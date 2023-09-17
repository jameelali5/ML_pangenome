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
# Loading all models metrics and joint them in a dictionary:
path = '/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/'
Model_Scores = {}

# %%
# Loading all models metrics and joint them in a dictionary:
path = '/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/'
Model_Scores = {}

# Loading all data frames
ecoli_GB_metrics = pd.read_csv(path+"GB_metrics_df.csv")
PA_GB_metrics = pd.read_csv(path+'PA_GB_metrics_df.csv')


# Adding all dataframes to dictionary
Model_Scores["ecoli_Gradient_Boosted_Trees"] = ecoli_GB_metrics
Model_Scores["PA_Gradient_Boosted_Trees"] = PA_GB_metrics

# %%
# Access dataframes
Model_Scores['PA_Gradient_Boosted_Trees'].head()

# %%
# Extracting list of combos from dictionary created
combo_list = list(Model_Scores['ecoli_Gradient_Boosted_Trees']["Drug_combo"].str.split("_", expand= True)[1].unique())
combo_list 

# %%
# Extracting list of drugs from dictionary created
# drug_list = list(Model_Scores['Gradient_Boosted_Trees']["Drug_combo"].str[:3].unique())
drug_list = ['CIP', 'CTZ', 'TBM' ]

# %%
'''Function that will take the best scores from each model'''
def Best_metrics(model,df):
  print("Selecting Best Scores for model: ",model)
  bestcores_dic = {}
  for drug in drug_list:
    data = df.loc[df["Drug_combo"].str.startswith(drug)]
    max_acc = max(data["Accuracy"])
    drug_combo = data["Drug_combo"][data["Accuracy"] == max_acc].unique()[0]
    R_recall = float(data[data["Drug_combo"] == drug_combo]["R_recall"])
    S_recall = float(data[data["Drug_combo"] == drug_combo]["S_recall"])
    bestcores_dic[drug_combo] = [max_acc, R_recall, S_recall]
  bestscores_df = pd.DataFrame.from_dict(bestcores_dic, orient ='index',columns=["Accuracy", "S_recall", "R_recall"]).reset_index()
  bestscores_df = bestscores_df.rename(columns = {'index':'Drug_combo'})
  return bestscores_df


# %%
'''Function for creating separate columns for feature combinations'''
def make_GYS(bestscores_df):
      # Create 3 new columns one for each type of features (Gene presence, Year and Population structure)
  bestscores_df["G"] = " "
  bestscores_df["Y"] = " "
  bestscores_df["S"] = " "

  # Read the combo part of Drug_combo
  split_c = bestscores_df["Drug_combo"].str.split("_", expand=True)
  i=0
  while i < len(split_c[1]):
    split_each_c = [x for x in split_c[1][i]]
    for g in split_each_c:
      if "G" in split_each_c:
        bestscores_df.at[i,"G"] = 1
      else:
        bestscores_df.at[i,"G"] = 0
    for y in split_each_c:
      if "Y" in split_each_c:
        bestscores_df.at[i,"Y"] = 1
      else:
        bestscores_df.at[i,"Y"] = 0
    for s in split_each_c:
      if "S" in split_each_c:
        bestscores_df.at[i,"S"] = 1
      else:
        bestscores_df.at[i,"S"] = 0
    i += 1
  bestscores_df["Drug_combo"] = bestscores_df["Drug_combo"].map(lambda x: x.rstrip('_GYS'))
  bestscores_df.rename(columns={"Drug_combo": "Drug"}, inplace = True)

  return bestscores_df


# %%
# Getting the best metrics for all the models
Best_metrics_models = {}
for model, df in Model_Scores.items():
  # select the best scores obtained from each model
  Model_best_metrics = Best_metrics(model, df)

  # Code GYS data in 0 "for absence" and 1 "for presence" into 3 columns
  GYS_coded_best = make_GYS(Model_best_metrics)
  print(GYS_coded_best)

  # Save new dataframe in a dictionary with best metrics selected
  Best_metrics_models[model] = GYS_coded_best

# %%
'''Function to create bars for plot'''
def barplot(metric_col, subplot_axis, label_show = True):
  for model, df in Best_metrics_models.items():
    X_axis = np.arange(len(drug_list))
    X_labels = drug_list

    Y_axis = np.arange(0,1.2,0.2)

    subplot_axis.set_ylabel(metric_col, fontsize = 25)
    subplot_axis.set_ylim(bottom=0, top=1)

    subplot_axis.set_xticklabels(X_labels, fontsize = 20)
    subplot_axis.set_xticks(X_axis)
    subplot_axis.margins(x=0)

    if label_show == False:
      subplot_axis.tick_params(left = True, right = False , labelleft = True , labelbottom = False, bottom = True)

    if model == "ecoli_Gradient_Boosted_Trees":
      X_axis = X_axis - 0.1
      color = "#0d3b66"
      label = "ecoli_GB"
      subplot_axis.bar(X_axis, list(df[metric_col]), width =.2, align = 'center', color = color, label = label, edgecolor="gray")
    elif model == "PA_Gradient_Boosted_Trees":
      X_axis = X_axis + 0.1
      color = "#faf0ca"
      label = "PA_GB"
      subplot_axis.bar(X_axis, list(df[metric_col]), width =.2, align = 'center', color = color, label = label, edgecolor="gray")

  return

# %%
'''Function to plot feature combinations'''
def GYS_gridplot(drug, subplot_axis, label_show = True, title_pos = -0.2):
      # Create 3 new lists one for each type of features (Year, Population structure and Gene presence)
  Y_list = [] # storing whether it used or not year data
  S_list = [] # storing whether it used or not population structure data
  G_list = [] # storing whether it used or not accessory gene data

  # Fill up corresponding lists from drug results from each model
  for model, df in Best_metrics_models.items():
    for drug_name in df["Drug"]:
      if drug_name == drug:
        Y_list.append(int(df["Y"][df["Drug"]==drug]))
        S_list.append(int(df["S"][df["Drug"]==drug]))
        G_list.append(int(df["G"][df["Drug"]==drug]))

  Drug_GYS_list = [Y_list, S_list, G_list]

  plt.yticks(np.arange(3), ["Year", "Structure", "Accessory Genes"])
  if label_show == False:
      subplot_axis.tick_params(left = False, labelleft = False)

  orig_map = plt.cm.get_cmap('gray')
  reversed_map = orig_map.reversed()
  subplot_axis.imshow(Drug_GYS_list, cmap = reversed_map, aspect = 0.4)

  subplot_axis.axvline(x=0.5)
  subplot_axis.axvline(x=1.5)
  subplot_axis.axhline(y=0.5)
  subplot_axis.axhline(y=1.5)
  subplot_axis.set_title(drug, fontsize= 25, y=title_pos)
  subplot_axis.tick_params(
    axis = 'x',
    which = 'both',
    bottom = False,
    top = False,
    labelbottom = False)
  return [Y_list, S_list, G_list]

# %%
# Code to create bargraphs
fig = plt.figure(figsize = (20,15), constrained_layout=False)

gs1 = fig.add_gridspec(nrows=4, ncols=3, left=0.05, right=0.5, wspace=0.07)
# Accuracy barcharts for all models
acc_axis = fig.add_subplot(gs1[0, :])
acc_axis.set_title('Prediction of Antibiotic Resistance From Gradient Boosted Trees', fontsize = 30, y = 1.2, weight = 'semibold')
acc_plot = barplot("Accuracy",acc_axis, label_show = False)

# R_f1_score barcharts for all models
R_metric_axis = fig.add_subplot(gs1[1, :])
R_metric_plot = barplot('R_recall',R_metric_axis, label_show = False) # We can swith to recall as well

# S_f1_score barcharts for all models
S_metric_axis = fig.add_subplot(gs1[2, :])
S_metric_plot = barplot('S_recall',S_metric_axis, label_show = False) # We can swith to recall as well

# GYS gridplot charts for each drug
gs2 = fig.add_gridspec(nrows=1, ncols=3, top=0.555, bottom=0,left=0.05, right=0.5, wspace=1.5)
i=0
while (i< len(drug_list)):
  for drug in drug_list:
    if i == 0:
      Drug_grid = fig.add_subplot(gs2[-1, i])
      drug_GYS = GYS_gridplot(drug, Drug_grid, label_show=True, title_pos=-0.5)
      i+=1
    else:
      Drug_grid = fig.add_subplot(gs2[-1, i])
      drug_GYS = GYS_gridplot(drug, Drug_grid, label_show=False, title_pos=-0.5)
      i+=1

legend_elements = [Patch(facecolor='#0d3b66', edgecolor='gray', label='E. coli'),
                   Patch(facecolor='#faf0ca', edgecolor='gray', label='P. aeruginosa'),
                   Patch(facecolor='black', edgecolor="black", label='Used'),
                   Patch(facecolor='white', edgecolor="black",label='Not Used')]

plt.legend(handles=legend_elements,loc = 'lower center', bbox_to_anchor=(-2, -1), prop ={'size': 20}, borderaxespad=-2, ncol = 2)

# %%
fig.savefig('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Images/combined_GB_plot.jpg', bbox_inches="tight")
# %%
