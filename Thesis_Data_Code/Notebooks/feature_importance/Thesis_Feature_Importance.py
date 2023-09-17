# %%
# Data manipulation imports for ML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import packages for Gradient Boosted Tree model
from xgboost import XGBClassifier

# Imports for model evaluation 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Imports for data visualization
import matplotlib.pyplot as plt


# %%
# Load csv file as a dataframe
path = '/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/'
All_Drugs_df = pd.read_csv(path+"EColi_Merged_dfs.csv", na_values="NaN")
All_Drugs_df


# %%
# Creating a list of antibiotic names
# We want all features to be used to train on individual drugs therefore the need to create individual drug dataframes
drug_list = All_Drugs_df.iloc[:,1:13].columns
drug_list


# %%
'''Function that makes dataframes for each antibiotic and dropping NaN values'''
def makeDF(drug):
  df_list = [All_Drugs_df[["Isolate",drug]],All_Drugs_df.iloc[:,13:]]
  Drug_df = pd.concat(df_list, axis=1)
  Drug_df = Drug_df.dropna()
  return Drug_df



# %%
'''Separating each dataframe into Labels and Features for training and testing data'''
def Split_train_test(Drug_df,drug): # Pass in drug df and drug name
  Train_test_dic = {}
  labels = Drug_df[drug] # Grab the S/R data from the df
  features = Drug_df.drop(columns=[drug]) # drop the drug col from the feature dataset
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42, stratify=labels)

  # Capture the split datasets into a dictionary
  Train_test_dic['labels_train'] = labels_train
  Train_test_dic['features_train'] = features_train
  Train_test_dic['labels_test'] = labels_test
  Train_test_dic['features_test'] = features_test

  return Train_test_dic

# %%
# Making a list of combinations of data sources we would like to test in our ML models
combo_list = ['G', 'S', 'GY', 'GS', 'SY', 'GYS' ] 
# combo_list = ['GYS' ] 
# G = Genes, S = Population Structure, Y = Year

# %%
'''Function that creates different feature combinations of the predictor features'''
def combo_feat(features_df, drug, combo): # Pass in feature df, 'drug name', data source combination 'GYS'

  # creating Year column filters for features_df
  year_filter = [col for col in features_df if col.startswith("Year")]
  year_feat = features_df[year_filter]

  # creating Population structure column filters for features_df
  pop_str_filter = [col for col in features_df if col.startswith("Cutoff")]
  pop_struc_feat = features_df[pop_str_filter]

  # creating Gene precence column filters for features_df
  gene_presc_filter = [col for col in features_df.columns if col not in pop_str_filter and col not in year_filter and col != "Isolate"]
  gene_presc_feat = features_df[gene_presc_filter]

  if combo == 'G':
    df_list = [features_df['Isolate'], gene_presc_feat]
    G_feat_df = pd.concat(df_list, axis=1)
    G_feat_df = G_feat_df.drop(columns=['Isolate'])
    return G_feat_df

  if combo == 'S':
    df_list = [features_df['Isolate'], pop_struc_feat]
    S_feat_df = pd.concat(df_list, axis=1)
    S_feat_df = S_feat_df.drop(columns=['Isolate'])
    return S_feat_df

  if combo == 'GY':
    df_list = [features_df['Isolate'], gene_presc_feat, year_feat]
    GY_feat_df = pd.concat(df_list, axis=1)
    GY_feat_df = GY_feat_df.drop(columns=['Isolate'])
    return GY_feat_df

  if combo == 'GS':
    df_list = [features_df['Isolate'], gene_presc_feat, pop_struc_feat]
    GS_feat_df = pd.concat(df_list, axis=1)
    GS_feat_df = GS_feat_df.drop(columns=['Isolate'])
    return GS_feat_df

  if combo == 'SY':
    df_list = [features_df['Isolate'], pop_struc_feat, year_feat]
    SY_feat_df = pd.concat(df_list, axis=1)
    SY_feat_df = SY_feat_df.drop(columns=['Isolate'])
    return SY_feat_df

  if combo == 'GYS':
    df_list = [features_df['Isolate'], gene_presc_feat, year_feat, pop_struc_feat, ]
    GYS_feat_df = pd.concat(df_list, axis=1)
    GYS_feat_df = GYS_feat_df.drop(columns=['Isolate'])
    return GYS_feat_df


# %%
# Let's check all drugs
drug_list

# %%
# Let's see all combinations we are interested in
combo_list

# %%
# create another empty dictionary to hold feature importance
GB_feat_imp = {}


# %%
'''Instantiating Gradient Boosted Trees model function'''
def run_GB(feat_train_df, lab_train, drug, combo, num_iter):
  print(drug +" Training combo: "+ combo)
  GB =  XGBClassifier(random_state = None, class_weight='balanced') # Completely randomize datasets
  GB = GB.fit(feat_train_df, lab_train)
  GB_feat_imp[drug+"_"+combo+'_'+str(num_iter)] = GB.get_booster().get_score(importance_type='gain')# Feature importance code
  return GB

# %%
'''Function using the model created and trained and the feature combinations from testing data'''
def predict(GB_combo_Model, features_test):
  labels_pred = GB_combo_Model.predict(features_test)
  return labels_pred


# %%
'''Function that evaluates our model using our actual and predicted data'''
'''def evaluate(GB_combo_model, labels_test, labels_pred, cf= True):
  report = classification_report(labels_test, labels_pred, output_dict = True)
  accuracy = report['accuracy']
  R_f1_score = report['R']['f1-score']# Resistant
  S_f1_score = report['S']['f1-score']# Susceptible
  if cf == True:
    cm = confusion_matrix(labels_test, labels_pred, labels=GB_combo_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GB_combo_model.classes_)
    disp.plot()
    plt.show()
  return [accuracy,R_f1_score,S_f1_score]'''

def evaluate(LG_combo_model, labels_test, labels_pred, cf= True, show_results=True):
  report = classification_report(labels_test, labels_pred, output_dict = True)
  if cf == True:
    cm = confusion_matrix(labels_test, labels_pred, labels=GB_combo_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GB_combo_model.classes_)
    disp.plot()
    plt.show()
  if show_results == True:
    print("Results")
    print('Accuracy:',report['accuracy'])
    print('R recall:',report['R']['recall'])
    print('S recall:',report['S']['recall'])
    print('R precision:',report['R']['precision'])
    print('S precision:',report['S']['precision'])
  return [report['accuracy'], report['R']['recall'], report['S']['recall'], report['R']['precision'], report['S']['precision']]


# %%
# Lets use all our functions this time and save our report into a single data structure
GB_model_metrics = {}
loop_iter = 0
num_iter = 0

while loop_iter < 50: # takes about 6 - 8 hours to run
  print ('Iteration:'+str(loop_iter))   
  for drug in drug_list:
    print(drug)
    Drug_df = makeDF(drug) # creates one df per drug
    Test_Train_dic = Split_train_test(Drug_df, drug) # splits each drug df into a dictionary with testing and training data
    for combo in combo_list:
      # Training each drug_combo features
      labels_train = Test_Train_dic["labels_train"]
      features_train = combo_feat(Test_Train_dic["features_train"], drug, combo) # create corresponding feature_df for training
      GB_combo_model = run_GB(features_train, labels_train, drug, combo, num_iter) # runs logistic regression model using the corresponding training feature_df 
      
      # Predicting each drug_combo features
      features_test = combo_feat(Test_Train_dic["features_test"], drug, combo) # create corresponding feature_df for testing
      labels_pred = predict(GB_combo_model, features_test) # generate predictions based on the feature combination tested

      # Evaluating our models
      labels_test = Test_Train_dic["labels_test"]
      report = evaluate(GB_combo_model, labels_test, labels_pred, cf=False, show_results=False)
      GB_model_metrics[drug+"_"+combo] = report
      
      if drug == drug_list[-1]: # For naming to save feature gain information
        num_iter = num_iter + 1

      print(report)
  
  loop_iter = loop_iter + 1 # increment while loop by one

# %%
# convert dicitionary holding feature importance into df
GB_feat_imp_df = pd.DataFrame.from_dict(GB_feat_imp, orient='index')

# %%
# Saving feat import into a CSV file
GB_feat_imp_df.to_csv(path+"GB_feat_imp_df_400_ecoli.csv") # for ecoli
# GB_feat_imp_df.to_csv(path+"GB_feat_imp_df_PA.csv") # for PA
# %%
