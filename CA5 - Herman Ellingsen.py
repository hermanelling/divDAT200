#!/usr/bin/env python
# coding: utf-8

# # DAT200 CA5 2022
# 
# Kaggle username: Herman Ellingsen

# ### Imports

# In[252]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import path

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# ### Reading data

# In[253]:


raw_train_data = pd.read_pickle("/Users/Herman/Documents/dat200-ca5-2022/train.pkl")
raw_test_data = pd.read_pickle("/Users/Herman/Documents/dat200-ca5-2022/test.pkl")


# ### Data exploration and visualisation

# In[254]:


raw_train_data.info()  # Meta data about the dataset


# In[265]:


raw_train_data.describe() # Meta data about the dataset
# Dont know why it only shows one column?


# In[266]:


raw_train_data.shape # Checking the shape of the data


# In[267]:


raw_train_data.head() # Print the first 5 rows to have a look at the data


# In[268]:


# Explore the missing values

print(raw_train_data.isin(['missing']).sum())

numerical_data = raw_train_data.drop(columns={"Season", "Weather situation"})
numerical_data = numerical_data.replace("missing", np.nan)


plt.figure(figsize=(10,6))
sns.heatmap(numerical_data.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
plt.title("Visualizing Missing Data")


# ### Data cleaning

# In[269]:


# Cleaning dataset using KNN-imputer:
numerical_data = raw_train_data.drop(columns={"Season", "Weather situation"})
numerical_data = numerical_data.replace("missing", np.nan)

imp_mean = KNNImputer(n_neighbors=3)
imp_mean.fit(numerical_data)
data = pd.DataFrame(imp_mean.transform(numerical_data), columns=numerical_data.columns)

samlet_data = pd.concat([data, raw_train_data[["Season", "Weather situation"]]], axis=1)

y_knnimputer = samlet_data["Rental bikes count"]
X_knnimputer = samlet_data.drop(columns={"Rental bikes count"})

X_knnimputer.isna().values.any()


# In[270]:


# Cleaning dataset using SimpleImputer:
numerical_data = raw_train_data.drop(columns={"Season", "Weather situation"})
numerical_data = numerical_data.replace("missing", np.nan)

imp_mean = SimpleImputer(strategy="mean")
imp_mean.fit(numerical_data)
data = pd.DataFrame(imp_mean.transform(numerical_data), columns=numerical_data.columns)

samlet_data = pd.concat([data, raw_train_data[["Season", "Weather situation"]]], axis=1)


y_simpleimputer = samlet_data["Rental bikes count"]
X_simpleimputer = samlet_data.drop(columns={"Rental bikes count"})

X_simpleimputer.isna().values.any()


# In[271]:


# Cleaning dataset by simply removing null values:
X2 = raw_train_data.replace("missing", np.nan)
X2 = X2.dropna()

y_dropna = X2["Rental bikes count"]
X_dropna = X2.drop(columns={"Rental bikes count"})

X_dropna.isna().values.any()


# ### Data exploration after cleaning

# In[272]:


# Checking for any missing values after cleaning
sns.heatmap(X_knnimputer.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})

print(X_knnimputer.isna().any())


# In[273]:


# Compare features to each other
# Kept the features that was most interesting to explore
sns.pairplot(data=X_knnimputer[["Temperature (normalized)", "Feels-like temperature (normalized)", "Month", "Humidity (normalized)"]], corner=True)


# In[274]:


# Use a heatmap to see the features correlation
# with each other
sns.heatmap(X_knnimputer.corr())


# In[275]:


# Use violinplot to show distribution of values in each feature
# After scaling the data, its easier to see distribution
data_4_violin_1 = X_knnimputer.drop(columns={"Season", "Weather situation", "Holiday", "Windspeed"})
data_4_violin_2 = X_knnimputer[["Holiday", "Windspeed"]]
scaler = StandardScaler()
data_4_violin_sc_1 = scaler.fit_transform(data_4_violin_1.select_dtypes(include='number'))
data_4_violin_sc_2 = scaler.fit_transform(data_4_violin_2.select_dtypes(include='number'))

for subset, col in zip([data_4_violin_sc_1, data_4_violin_sc_2], 
                       [data_4_violin_1, data_4_violin_2]):
    f, ax = plt.subplots(figsize=(11, 6))
    sns.violinplot(data=subset) 
    sns.despine(left=True, bottom=True)
    plt.xticks(ticks=range(0, len(col.columns)), labels=col.columns, rotation=30)
    plt.title("Violin Plot")
    plt.show()


# ### Data preprocessing

# In[283]:


# Preprocessing class to be used in pipeline
class Preprocessing(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        
        X_testern = X.copy().dropna()
        
        X_testern = X_testern.drop(columns={"Windspeed"})
        X_testern["Year_sqrt"] = np.sqrt(X_testern["Year"]+1)
        X_testern["Year_log"] = np.log(X_testern["Year"]+1)
        X_testern["Temperature (normalized)_log"] = np.log(X_testern["Temperature (normalized)"]+1)
        X_testern["Humidity (normalized)_log"] = np.log(X_testern["Humidity (normalized)"]+1)
        X_testern["Year"] = (X_testern["Year"]+1)**2 # Remove?
        X_testern["Year_3"] = (X_testern["Year"]+1)**3
        X_testern["Year_4"] = (X_testern["Year"]+1)**4
        X_testern["Hour * Working day"] = (X_testern["Hour"]) * X_testern["Working day"]
        X_testern["Hour * Month"] = X_testern["Hour"] * X_testern["Month"]
        X_testern["hum/hol"] = ((X_testern["Humidity (normalized)"]+1) / (X_testern["Holiday"]+1)**2)
        
        
        X_testern["Working day * Holiday"] = (X_testern["Working day"]+1) / (X_testern["Holiday"]+1)
        X_testern["Month * Holiday"] = (X_testern["Month"]+1) * (X_testern["Holiday"]+1)
        X_testern["Month * Humidity"] = (X_testern["Month"]+1) / (X_testern["Humidity (normalized)"]+1)
        
        return X_testern


# ### Modelling

# #### Data pipeline with regression model

# In[284]:


for X, y, imp in zip([X_knnimputer, X_simpleimputer, X_dropna],
                     [ y_knnimputer, y_simpleimputer, y_dropna],
                     ["KNNImputer()",  "SimpleImputer", "dropna()"]):

    
    # Here I concatinate the train and test set before I make dummy variables.
    # This is because I got different amount of dummy variables if I
    # did this seperate
    train_objs_num = len(X)
    dataset = pd.concat(objs=[X, raw_test_data], axis=0)
    dataset_preprocessed = pd.get_dummies(dataset)
    train_preprocessed = dataset_preprocessed[:train_objs_num]
    test_preprocessed = dataset_preprocessed[train_objs_num:]


    pipe_rf = make_pipeline(Preprocessing(),
                            StandardScaler(),
                            RandomForestRegressor(random_state=2,
                                                  n_jobs=-1))



    # Set the value range for the GridSearch
    # I tried different value ranges, but did not have time
    # to run through the whole thing again. So this only
    # a short version with the hyperparamters I
    # used in the final submission
    param_grid_rf = [{'randomforestregressor__n_estimators': [350],
                      'randomforestregressor__max_depth': [25],
                      'randomforestregressor__min_samples_leaf': [1], 
                      'randomforestregressor__min_samples_split': [2],
                      'randomforestregressor__min_impurity_decrease': [0.0035],
                      'randomforestregressor__max_features': [0.57]}]



    # Create the GridSearch
    gs_rf = GridSearchCV(estimator=pipe_rf, 
                          param_grid=param_grid_rf,
                          scoring='r2',
                          cv=5,
                          refit=True,
                          n_jobs=-1)


    # Fit and predict
    gs_rf.fit(train_preprocessed, y)

    # Print the best scores and the hyperparameters that gave the best results
    print("-"*60)
    print(f"R2 score and best parameters when using {imp}")
    print("-"*60)
    print(f"R2 Score: {gs_rf.best_score_}\n")
    print("{:<50} {:<15}\n".format("Hyperparameter", "Value"))
    for key, value in gs_rf.best_params_.items():
        print(f"{key:<50} {value:<15}")
    print(" ")
    print(" ")
    #estimator.get_params().keys()
    #clf.named_steps["preprocessing"].transform(X_train)


# #### Data pipeline with classification model

# Binning train target values
# 
# Can be performed with ex. pandas.qcut or pandas.cut
# 
# ```python
# n_bins = 10
# y_train_binned = pd.cut(y_train, n_bins, labels=False) # or
# y_train_binned = pd.qcut(y_train, n_bins, labels=False) 
# ```

# In[279]:


n_bins = range(2, 21, 2)
results = []


for bins in n_bins:
    
    y = samlet_data["Rental bikes count"]
    y_binned = pd.qcut(y, bins, labels=False) 
    X = samlet_data.drop(columns={"Rental bikes count"})
    X_testern = pd.get_dummies(X.copy())

    X_train, X_test, y_train, y_test = train_test_split(X_testern, y_binned, test_size=0.20, random_state=2)


    # Make pipeline for model
    pipe_classification = make_pipeline(Preprocessing(),
                                        RandomForestClassifier(random_state=2))

    pipe_classification.fit(X_train, y_train)

    results.append(pipe_classification.score(X_test, y_test))


# In[280]:


f, ax = plt.subplots(figsize=(11, 6))
plt.plot(n_bins, results, "bo--")
plt.title("Accuracy when binning train target values")
plt.xlabel("Number of bins")
plt.xticks(ticks=n_bins, labels=n_bins)
plt.ylabel("Mean accuracy")
plt.grid()

axes = plt.gca()
axes.set_ylim([0.30, 1])

for x, y in zip(n_bins, results):
    label = "{:.2f}".format(y)

    plt.annotate(label,
                 (x, y),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='left')


# As we see from the plot above, the mean accuracy decreases with the number of bins. I believe this is because the classifier needs to be more accurate when the number of target classes increases, therefore making it harder to get a high accuracy.
# 
# Note: For simplicity the test above was done on one train_test_split. The the "true" result might therefore be splightly different. 

# ### Kaggle submission

# In[281]:


def submission_file(y_pred):
    pathern = "/Users/Herman/Documents/CA5_submissions/CA5_01.csv"
    if not path.exists(pathern):
        pass
    else:   
        while path.exists(pathern):
            pathern = "/Users/Herman/Documents/CA5_submissions/CA5_"+str(int(pathern[-6:-4])+1).zfill(2)+".csv"
            print(pathern)
        
    CA5_sub = pd.DataFrame() # Make empty dataframe for submission
    y_pred_df = pd.DataFrame(y_pred) 
    CA5_sub["idx"] = y_pred_df.index # Insert index into the submission df
    CA5_sub["Rental bikes count"] = y_pred # Insert the predictions into the submission df
    CA5_sub.to_csv(pathern, index=None) # Convert dataframe into csv-file
    test_csv = pd.read_csv(pathern) # Checking if its in the right format
    print(test_csv)


# In[282]:


y_pred = gs_rf.predict(pd.get_dummies(test_preprocessed))
submission_file(y_pred)

