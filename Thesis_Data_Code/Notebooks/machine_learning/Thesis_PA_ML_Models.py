# %%
# Data manipulation imports for ML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import packages for logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Import packages for Random Forest model
from sklearn.ensemble  import RandomForestClassifier

# Import packages for Gradient Boosted Tree model
from xgboost import XGBClassifier

# Imports for model evaluation 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Imports for data visualization
import matplotlib.pyplot as plt

######################################### Upload combined df


# %%
# Load csv file as a dataframe
path = '/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data/'
All_Drugs_df = pd.read_csv(path+"PA_Merged_dfs.csv", na_values="NaN")
All_Drugs_df

#%%
All_Drugs_df.shape

#%%
All_Drugs_df['TBM'].value_counts(normalize=True)
#%%
All_Drugs_df['CTZ'].value_counts(normalize=True)
#%%
All_Drugs_df['CIP'].value_counts(normalize=True)

######################################## Create a single df for each drug

# %%
# Creating a list of antibiotic names
# We want all features to be used to train on individual drugs therefore the need to create individual drug dataframes
drug_list = All_Drugs_df.iloc[:,1:6].columns
drug_list

# %%
'''Function that makes dataframes for each antibiotic and dropping NaN values'''
def makeDF(drug):
  df_list = [All_Drugs_df[["Isolate",drug]],All_Drugs_df.iloc[:,6:]]
  Drug_df = pd.concat(df_list, axis=1)
  Drug_df = Drug_df.dropna()
  return Drug_df


# %%
'''Separating each dataframe into Labels and Features for training and testing data'''
def Split_train_test(Drug_df,drug): # Pass in drug df and drug name
  Train_test_dic = {}
  labels = Drug_df[drug] # Grab the S/R data from the df
  features = Drug_df.drop(columns=[drug]) # drop the drug col from the feature dataset
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)

  # Capture the split datasets into a dictionary
  Train_test_dic['labels_train'] = labels_train
  Train_test_dic['features_train'] = features_train
  Train_test_dic['labels_test'] = labels_test
  Train_test_dic['features_test'] = features_test

  return Train_test_dic


# %%
# Making a list of combinations of data sources we would like to test in our ML models
combo_list = ['G', 'S', 'GS'] 
# G = Genes, S = Population Structure, Y = Year

# %%
'''Function that creates different feature combinations of the predictor features'''
def combo_feat(features_df, drug, combo): # Pass in feature df, 'drug name', data source combination 'GYS'

  # creating Population structure column filters for features_df
  pop_str_filter = [col for col in features_df if col.startswith("Cutoff")]
  pop_struc_feat = features_df[pop_str_filter]

  # creating Gene precence column filters for features_df
  gene_presc_filter = [col for col in features_df.columns if col not in pop_str_filter and col != "Isolate"]
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

  if combo == 'GS':
    df_list = [features_df['Isolate'], gene_presc_feat, pop_struc_feat]
    GS_feat_df = pd.concat(df_list, axis=1)
    GS_feat_df = GS_feat_df.drop(columns=['Isolate'])
    return GS_feat_df
  
################################################ Logistic Regression Model


# %%
# Creating Logistic regression model function
@ignore_warnings(category=ConvergenceWarning)
def run_LG(feat_train_df, lab_train, drug, combo):
  print(drug +" Training combo: "+ combo)
  LG = LogisticRegression(random_state = 42, solver= 'lbfgs', C=1.0, max_iter=500, class_weight='balanced') 
  LG = LG.fit(feat_train_df, lab_train)
  return LG

# %%
'''Function using the model created and trained and the feature combinations from testing data'''
def predict(LG_combo_Model, features_test):
  labels_pred = LG_combo_Model.predict(features_test)
  return labels_pred

# %%
'''Function that evaluates our model using our actual and predicted data'''
'''def evaluate(LG_combo_model, labels_test, labels_pred, cf= True):
  report = classification_report(labels_test, labels_pred, output_dict = True)
  accuracy = report['accuracy']
  R_f1_score = report['R']['f1-score']# Resistant
  S_f1_score = report['S']['f1-score']# Susceptible
  if cf == True:
    cm = confusion_matrix(labels_test, labels_pred, labels=LG_combo_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LG_combo_model.classes_)
    disp.plot()
    plt.show()
  return [accuracy,R_f1_score,S_f1_score]'''

# Creating a function that evaluates our model using our actual and predicted data
def evaluate(LG_combo_model, labels_test, labels_pred, cf= True, show_results=True):
  report = classification_report(labels_test, labels_pred, output_dict = True)
  if cf == True:
    cm = confusion_matrix(labels_test, labels_pred, labels=LG_combo_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LG_combo_model.classes_)
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
# Let's check all drugs
drug_list

# %%
# Let's see all combinations we are interested in
combo_list

# %%
# Logistic Regression - All combinations
LG_model_metrics = {}

for drug in drug_list:
  print(drug)
  Drug_df = makeDF(drug) # creates one df per drug
  Test_Train_dic = Split_train_test(Drug_df, drug) # splits each drug df into a dictionary with testing and training data
  for combo in combo_list:
    # Training each drug_combo features
    labels_train = Test_Train_dic["labels_train"]
    features_train = combo_feat(Test_Train_dic["features_train"], drug, combo) # create corresponding feature_df for training
    LG_combo_model = run_LG(features_train, labels_train, drug, combo) # runs logistic regression model using the corresponding training feature_df 
    
    # Predicting each drug_combo features
    features_test = combo_feat(Test_Train_dic["features_test"], drug, combo) # create corresponding feature_df for testing
    labels_pred = predict(LG_combo_model, features_test) # generate predictions based on the feature combination tested

    # Evaluating our models
    labels_test = Test_Train_dic["labels_test"]
    report = evaluate(LG_combo_model, labels_test, labels_pred, cf=False, show_results=False)
    LG_model_metrics[drug+"_"+combo] = report
    
    print(report)

# %%
# Convert LG dictionary into a dataframe
LG_metrics = pd.DataFrame.from_dict(LG_model_metrics, orient='index',columns=["Accuracy", "R_recall", "S_recall", "R_precision", "S_precision"]).reset_index()
LG_metrics = LG_metrics.rename(columns = {'index':'Drug_combo'})

# Saving LG df into a CSV file
LG_metrics.to_csv(path+"PA_LG_metrics_df.csv", index= False)
LG_metrics


############################################## Random Forest Model


# %%
'''Creating Random Forest model function'''
def run_RF(feat_train_df, lab_train, drug, combo):
  print(drug +" Training combo: "+ combo)
  RF = RandomForestClassifier(random_state = 42, class_weight='balanced')
  RF = RF.fit(feat_train_df, lab_train)
  return RF

# %%
'''Function using the model created and trained and the feature combinations from testing data'''
def predict(RF_combo_Model, features_test):
  labels_pred = RF_combo_Model.predict(features_test)
  return labels_pred

# %%
'''Function that evaluates our model using our actual and predicted data'''
'''def evaluate(RF_combo_model, labels_test, labels_pred, cf= True):
  report = classification_report(labels_test, labels_pred, output_dict = True)
  accuracy = report['accuracy']
  R_f1_score = report['R']['f1-score']# Resistant
  S_f1_score = report['S']['f1-score']# Susceptible
  if cf == True:
    cm = confusion_matrix(labels_test, labels_pred, labels=RF_combo_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RF_combo_model.classes_)
    disp.plot()
    plt.show()
  return [accuracy,R_f1_score,S_f1_score]'''

# Creating a function that evaluates our model using our actual and predicted data
def evaluate(LG_combo_model, labels_test, labels_pred, cf= True, show_results=True):
  report = classification_report(labels_test, labels_pred, output_dict = True)
  if cf == True:
    cm = confusion_matrix(labels_test, labels_pred, labels=RF_combo_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RF_combo_model.classes_)
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
# Random Forest - All combinations of features
RF_model_metrics = {}

for drug in drug_list:
  print(drug)
  Drug_df = makeDF(drug) # creates one df per drug
  Test_Train_dic = Split_train_test(Drug_df, drug) # splits each drug df into a dictionary with testing and training data
  for combo in combo_list:
    # Training each drug_combo features
    labels_train = Test_Train_dic["labels_train"]
    features_train = combo_feat(Test_Train_dic["features_train"], drug, combo) # create corresponding feature_df for training
    RF_combo_model = run_RF(features_train, labels_train, drug, combo) # runs logistic regression model using the corresponding training feature_df 
    
    # Predicting each drug_combo features
    features_test = combo_feat(Test_Train_dic["features_test"], drug, combo) # create corresponding feature_df for testing
    labels_pred = predict(RF_combo_model, features_test) # generate predictions based on the feature combination tested

    # Evaluating our models
    labels_test = Test_Train_dic["labels_test"]
    report = evaluate(RF_combo_model, labels_test, labels_pred, cf=False, show_results=False)
    RF_model_metrics[drug+"_"+combo] = report
    
    print(report)


# %%
# Convert  RF dictionary into a dataframe
RF_metrics = pd.DataFrame.from_dict(RF_model_metrics, orient='index',columns=["Accuracy", "R_recall", "S_recall", "R_precision", "S_precision"]).reset_index()
RF_metrics = RF_metrics.rename(columns = {'index':'Drug_combo'})

# Save RF df into a CSV file
RF_metrics.to_csv(path+"PA_RF_metrics_df.csv", index= False)
RF_metrics


################################################### Gradient Boosted Tree

# %%
'''creating Gradient Boosted Trees model function'''
def run_GB(feat_train_df, lab_train, drug, combo):
  print(drug +" Training combo: "+ combo)
  GB =  XGBClassifier(random_state = 42, class_weight='balanced')
  GB = GB.fit(feat_train_df, lab_train)
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

# Creating a function that evaluates our model using our actual and predicted data
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
# create another empty dictionary to hold feature importance
GB_feat_imp = {}

for drug in drug_list:
  print(drug)
  Drug_df = makeDF(drug) # creates one df per drug
  Test_Train_dic = Split_train_test(Drug_df, drug) # splits each drug df into a dictionary with testing and training data
  for combo in combo_list:
    # Training each drug_combo features
    labels_train = Test_Train_dic["labels_train"]
    features_train = combo_feat(Test_Train_dic["features_train"], drug, combo) # create corresponding feature_df for training
    GB_combo_model = run_GB(features_train, labels_train, drug, combo) # runs logistic regression model using the corresponding training feature_df 
    
    # Predicting each drug_combo features
    features_test = combo_feat(Test_Train_dic["features_test"], drug, combo) # create corresponding feature_df for testing
    labels_pred = predict(GB_combo_model, features_test) # generate predictions based on the feature combination tested

    # Evaluating our models
    labels_test = Test_Train_dic["labels_test"]
    report = evaluate(GB_combo_model, labels_test, labels_pred, cf=False, show_results=False)
    GB_model_metrics[drug+"_"+combo] = report
  
    print(report)

# %%
# Convert GB dictionary into a dataframe
GB_metrics = pd.DataFrame.from_dict(GB_model_metrics, orient='index',columns=["Accuracy", "R_recall", "S_recall", "R_precision", "S_precision"]).reset_index()
GB_metrics = GB_metrics.rename(columns = {'index':'Drug_combo'})


# Saving GBresults into a CSV file
GB_metrics.to_csv(path+"PA_GB_metrics_df.csv", index= False)
GB_metrics

# %%
