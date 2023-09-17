# %%

# import libraries

# data wrangling
import pandas as pd
import numpy as np
from functools import reduce

# Machine Learning Model and Evaluation Imports
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier
from sklearn import metrics #Import for accuracy calculation
from sklearn.metrics import classification_report

# Tree Visualization Imports
from matplotlib import pyplot as plt
import graphviz
from sklearn import tree

# %%
# read in the csv
metadata_url = 'https://raw.githubusercontent.com/DaneshMoradigaravand/PanPred/master/test_data/Metadata.csv'
gene_presence_url = 'https://raw.githubusercontent.com/DaneshMoradigaravand/PanPred/master/test_data/AccessoryGene.csv'
pop_struc_labels_url = 'https://raw.githubusercontent.com/DaneshMoradigaravand/PanPred/master/test_data/PopulationStructure.csv_labelencoded.csv'

metadata = pd.read_csv(metadata_url)
gene_presence_data = pd.read_csv(gene_presence_url)
pop_struc_data = pd.read_csv(pop_struc_labels_url)


# %%
# check the metadata df
# 1936 rows
metadata.head()

# %%
# check the number of missing data from the metadata
metadata.isnull().sum()

# %%
# check the gene presence df
gene_presence_data.head()

# %%
# check the numbed of missing data from gene presence
gene_presence_data.isnull().sum()

# %%
# check the pop struc data
pop_struc_data.head()

# %%
# check the number of missing data from the pop struc data
pop_struc_data.isnull().sum()

# %%
# rename the first column
gene_presence_data = gene_presence_data.rename(columns={'Unnamed: 0' : 'Isolate'})
pop_struc_data = pop_struc_data.rename(columns={'Unnamed: 0' : 'Isolate'})

# %%
# check the headers for gene presence
gene_presence_data.head()

# %%
# check the headers for pop struc
pop_struc_data.head()

# %%
#def makeDF(requestedDF, drug_name, metadata, gene_presence_data, pop_struc_data):
def makeDF(requestedDF, drug_name):   
  
  # drug label data = isolateID + drug label
  drug_label = metadata # make a copy of metadata df
  drug_label = metadata[['Isolate', drug_name]] # only keep these certain columns

  # year data = year only
  year = metadata 
  year = year[['Isolate','Year']] 

  # list of sub dataframes (in order: gene presence-absence only, population structure only, gene + year, population + year, gene + pop + year)
  subDF_list = ['G', 'S', 'GY', 'SY', 'GYS' ] 

  # check which df you requested
  if requestedDF == 'G':
    merged_df = pd.merge(drug_label, gene_presence_data)
  elif requestedDF == 'S':
    merged_df = pd.merge(drug_label, pop_struc_data)
  elif requestedDF == 'GY':
    df_list = [drug_label, year, gene_presence_data] #list of dataframes to merge
    merged_df = reduce(lambda  left,right: pd.merge(left,right,on=['Isolate'], how='outer'), df_list) #if merging more than one df, must use reduce function
  elif requestedDF == 'SY':
    df_list = [drug_label, year, pop_struc_data] 
    merged_df = reduce(lambda  left,right: pd.merge(left,right,on=['Isolate'], how='outer'), df_list)
  elif requestedDF == 'GYS':
    df_list = [drug_label, year, gene_presence_data, pop_struc_data] 
    merged_df = reduce(lambda  left,right: pd.merge(left,right,on=['Isolate'], how='outer'), df_list)
  else: print('Unable to detect requested DF. Please input one of these as a string: G, S, GY, SY, GYS')
  
  return merged_df

# %%
#GRADIENT BOOSTED TREE   
class GB:
    # initialize method to create acc and report variables - necessary to run other functions
    def __init__(self, acc=None, report=None):
      self.acc = acc
      self.report = report
      self.feat_names = []
      self.feat_imp = []
      self.feat_dicts = []
    # function that initializes and trains a deciscion tree for the current drug 
    def run_GB(self, drug_name, i):
      DF_list = ['G', 'S', 'GY', 'SY', 'GYS']
      # load data
      data = makeDF(DF_list[i], drug_name) # call the makeDF function we made to create our df
      data = data.dropna() # drop rows with na values
      # assign labels and features
      labels = np.array(data[drug_name])
      features = data.drop(columns=[drug_name, 'Isolate']) 
      # split dataset
      features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42) # 70% training and 30% test
      # create decision tree classifier object
      gb = XGBClassifier(random_state = 42)
      # train decision tree classifier
      gb = gb.fit(features_train,labels_train)
      # create prediction label
      labels_pred = gb.predict(features_test)
      # calculate accuracy
      acc = round(100 * metrics.accuracy_score(labels_test, labels_pred),2)
      #print(drug_name," Accuracy:",acc,"%")
      # initialize classification report
      report = classification_report(labels_test, labels_pred, output_dict = True)
      if DF_list[i] == 'GYS': # want to get the feature importance for all features in the dataset
        # create temp variable to hold feature importance and append to feat_imp
        temp_feat_imp = gb.feature_importances_
        # create temp variable to hole the feature names and append to feat_names
        temp_feat_names = list(features)
        self.feat_dicts.append(dict(zip(temp_feat_names, temp_feat_imp)))
      return acc, report

    def make_GYS(self, DF_list, G_list, Y_list, S_list, i):
      # if else statement that checks for data type presence
      if DF_list[i] == 'G':
        G_list.append(1)
        Y_list.append(0) 
        S_list.append(0)
      elif DF_list[i] == 'S':
        G_list.append(0)
        Y_list.append(0) 
        S_list.append(1)
      elif DF_list[i] == 'GY':
        G_list.append(1)
        Y_list.append(1) 
        S_list.append(0)
      elif DF_list[i] == 'SY':
        G_list.append(0)
        Y_list.append(1) 
        S_list.append(1)
      elif DF_list[i] == 'GYS':
        G_list.append(1)
        Y_list.append(1) 
        S_list.append(1)
      else: print('Unable to detect requested DF. Please input one of these as a string: G, S, GY, SY, GYS')
    # function that calls the run_dt method and saves the evaluation metrics for every drug and df type combo

    def main_GB(self):
      # list of all our drugs we want to run a DT with
      #drug_list = ["CTZ", "CTX", "AMP", "AMX", "AMC", "TZP", "CXM", "CET", "GEN", "TBM", "TMP", "CIP"] 
      drug_list = ['CTZ']
      DF_list = ['G', 'S', 'GY', 'SY', 'GYS']
      # lists to store data type presence
      G_list, Y_list, S_list = [], [], []
      # list to store various evaluation metric values
      accuracy_list, precisionR_list, precisionS_list, recallR_list, recallS_list, f1scoreR_list, f1scoreS_list = [], [], [], [], [], [], []
      # list to store labels to index the table and list to store the drug names to append into the dataframe output
      drug_of_int_list = []
      # start at the first dataframe type
      i = 0
      while i < len(DF_list):
        # for loop that goes through each of the drug dfs
        for i in range(len(DF_list)):
          for drug_name in drug_list: # goes through each of the drugs and trains a dt
            acc, report = self.run_GB(drug_name, i) # calls the run_GB function to get the accuracies and evaluation reports of each DT model
            accuracy_list.append(acc)
            drug_of_int_list.append(drug_name)
            self.make_GYS(DF_list, G_list, Y_list, S_list, i) # calls the make_GYS function to create GYS columns
            for drug_class, drug_parameter in report.items(): # Returns the key:value pair of the nested dictionary for the classification report
              if drug_class == "S" or drug_class == "R": # Access only the Resistance and Susceptibility dictionaries from the classification report
                #print ("\ndrug_class:", drug_class)
                if drug_class == "S": # append labels to a list that will used to create dataframe
                  precisionS_list.append(round(report['S']['precision'],3))
                  recallS_list.append(round(report['S']['recall'],3))
                  f1scoreS_list.append(round(report['S']['f1-score'],3))
                else:
                  precisionR_list.append(round(report['R']['precision'],3))
                  recallR_list.append(round(report['R']['recall'],3))
                  f1scoreR_list.append(round(report['R']['f1-score'],3))
        # iterate to the next dataframe type
        i = i + 1
        # Create an array by combining all of our lists
        clr_combined = {'Drug': drug_of_int_list, 'G': G_list, 'Y': Y_list, 'S': S_list, 'Accuracy': accuracy_list, 'Precision_R': precisionR_list, 'Precision_S': precisionS_list, 'Recall_R' : recallR_list, 'Recall_S' : recallS_list, 'F1-Score_R': f1scoreR_list,'F1-Score_S': f1scoreS_list}
        # create data frame from the the array
        clr_comb_df = pd.DataFrame (clr_combined)

    def Get_Feature_Importance(self):
        return self.feat_dicts

# %%
# create instance of GB class and call the main function to get the complete df
GB = GB()
GB_df = GB.main_GB() # 25 min to run 

# %%
test = GB.Get_Feature_Importance()

# %%
test

# %%
len(test)
# %%
#test_df = pd.DataFrame.from_dict(test, orient='index')
# %%
#test_df 
# %%
