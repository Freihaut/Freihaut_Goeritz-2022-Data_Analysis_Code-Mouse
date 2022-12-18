'''
Code to run the machine learning analysis for the mouse-task data
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''

# import the relevant packages
import json
import pickle
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.inspection import permutation_importance

# Create custom classes for data transformation in the sklearn pipeline
from sklearn.base import TransformerMixin, BaseEstimator

# variance inflation factor calculation modules for multicollinearity handling
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant

# define a rng variable to ensure that the results are always the same
rng = np.random.RandomState(0)

# %%

# import the dataset
dataset = pd.read_csv("Mouse-Task_Analysis/Mouse_Task_Features.csv")

# transform the ID column from strings to labels
dataset["ID"] = LabelEncoder().fit_transform(dataset["ID"])


# %%

# list all potential predictors
all_predictors = ['task_duration', 'clicks', 'task_total_dist', 'task_speed_mean',
                  'task_speed_sd', 'task_abs_accel_mean', 'task_abs_accel_sd', 'task_abs_jerk_mean',
                  'task_abs_jerk_sd', 'task_angle_mean', 'task_angle_sd', 'task_x_flips', 'task_y_flips',
                  'trial_mean_duration', 'trial_sd_duration', 'trial_mean_trial_move_offset',
                  'trial_sd_trial_move_offset', 'trial_mean_total_dist', 'trial_sd_total_dist',
                  'trial_mean_distance_overshoot', 'trial_sd_distance_overshoot', 'trial_mean_speed_mean',
                  'trial_sd_speed_mean', 'trial_mean_speed_sd', 'trial_sd_speed_sd', 'trial_mean_abs_accel_mean',
                  'trial_sd_abs_accel_mean', 'trial_mean_abs_accel_sd', 'trial_sd_abs_accel_sd',
                  'trial_mean_abs_jerk_mean', 'trial_sd_abs_jerk_mean', 'trial_mean_abs_jerk_sd',
                  'trial_sd_abs_jerk_sd', 'trial_mean_angle_mean', 'trial_sd_angle_mean', 'trial_mean_angle_sd',
                  'trial_sd_angle_sd', 'trial_mean_x_flips', 'trial_sd_x_flips', 'trial_mean_y_flips',
                  'trial_sd_y_flips']

# list all covariates
covariates = ['order', 'zoom', 'screen_width', 'screen_height', 'age', 'sex', 'hand']

# list all targets
all_targets = ["arousal", "valence"]

# %%

############################################################
# Step 1: Split the dataset into training and testing data #
############################################################
# The dataset is split in a way that the first 80% of the completed measurement trials of each participant represent
# the training dataset and the last 20% of the completed measures represent the testing dataset. This resembles
# the application case that "new data" is used for prediction in an application case

# first, split the entire dataset in data subsets for each participant
split_dataframes = [y for x, y in dataset.groupby("ID", as_index=True)]

train_dfs = []
test_dfs = []
# iterate the dataframe of each participant
# measurements
for df in split_dataframes:
    # sort it by the timestamp column to order the trials chronologically
    df = df.sort_values("timestamp").reset_index()
    # add an order column
    df["order"] = range(len(df))
    # get the first 80% of the dataset
    df_80 = df.head(int(len(df) * .8))
    # get the remaining 20% of the dataset
    df_20 = df.iloc[max(df_80.index) + 1:]
    # add the participant train and test datasets to the list of training and test dfs
    train_dfs.append(df_80)
    test_dfs.append(df_20)

# combine the training and test datasets of each participant to create the final training and test dataset with data
# from all participants
train_df, test_df = pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)

#%%

# Get some basic information about the dataset
print(f"Shape of the entire dataset: {dataset.shape}")
print(f"Shape of the training dataset: {train_df.shape}")
print(f"Shape of the test dataset: {test_df.shape}")

#%%

# get some information about the number of trials per participant
print(f"Trials per participant for the entire dataset:\n{dataset['ID'].value_counts()}")
print(f"Number of participants in the entire dataset: {len(dataset['ID'].unique())}")
print("\n")
print(f"Trials per participant for the training dataset:\n{train_df['ID'].value_counts()}")
print(f"Number of participants in the training dataset: {len(train_df['ID'].unique())}")
print("\n")
print(f"Trials per participant for the test dataset:\n{test_df['ID'].value_counts()}")
print(f"Number of participants in the test dataset: {len(test_df['ID'].unique())}")


# %%

#################################################
# 2. Establish a dataset preprocessing pipeline #
#################################################
# Before the Machine Learning Algorithm is trained with the data (and its performance it tested on the test dataset),
# the dataset is preprocessed. This preprocessing includes multiple steps. Similar to the mouse data mixed model analysis,
# we apply multiple preprocessing routinesto the data to get multiple prediction results.
# This procedure also helps to get a sense for the robustness (variability) of the prediction results and therefore
# about the potential validity of the results

# The preprocessing includes:
# - Removing Outlier datasets (per mouse usage task)
# - Linear Equation of the mouse usage features to account for potential differences between the  mouse usage tasks
# - Removal of multicollinear mouse usage features

# In order to prevent data leakage in the machine learning analysis, all preprocessing steps are done using the
# training dataset and then applied to the testing dataset. We use the scikit-learn preprocessing pipeline for
# handling the preprocessing

# -----------------------------
# Preprocessing helper classes
# -----------------------------

# Custom scikit-learn classes to fit and transform training-test data with the specified transformations
# Most of these classes are rather specific. There might be better, more general solutions to the custom preprocessing
# steps in order to create different train-test dataset pairs

# A simple debugger class to get information about the data inside the pipeline (can be customized)
class DebugPipeline(BaseEstimator, TransformerMixin):

    def transform(self, X):
        # return the data
        print(X.shape)
        return X

    def fit(self, X, y=None, **fit_params):
        return self


# A custom transformer to process mulitcollinearity between features
class HandleMulticollinearity(BaseEstimator, TransformerMixin):
    """
    handle multicollinearity among independent variables based on three methods:
    1: Ignore = Use all independent variables
    2: Variance Inflation Factor: Use the VIF to stepwise remove independent variables until a threshold is met
    3: Correlation: Remove all independent variables with a correlation higher than a threshold
    """

    def __init__(self, method="select_all", cols=None, vif_thresh=10, cor_thresh=.80):
        self.method = method
        self.cols = cols
        self.vif_thresh = vif_thresh
        self.cor_thresh = cor_thresh
        self.sel_columns = []

    def fit(self, X, y=None):

        if self.method == "select_all":
            # dont drop any columns
            return self

        # stepwise remove variables based on their variance inflation factor
        elif self.method == "vlf":
            cols_to_delete = []
            # add a constant to the dataframe to calculate the vif
            X = add_constant(X.loc[:, self.cols])
            # calculate the variance inflation factor for all variables in the dataset and drop the feature with the
            # highest vif value. Repeat this process until no vif is greater than 10 (the specified threshold)
            while True:
                vlfs = pd.Series([vif(X.values, i) for i in range(X.shape[1])], index=X.columns)
                max_vlf = vlfs[1:].max()
                idx = vlfs[1:].idxmax()
                # print(max_vlf, idx)
                if max_vlf >= self.vif_thresh:
                    cols_to_delete.append(idx)
                    X.drop(idx, axis=1, inplace=True)
                else:
                    break
            # specify the columns that re deleted
            self.sel_columns = cols_to_delete
            return self

        # remove all variables with a correlation higher than a selected threshold
        elif self.method == "corr":
            # get the correlation matrix of the dataset
            corrmat = X.loc[:, self.cols].corr().abs()
            # get the average correlation of each column
            average_corr = corrmat.abs().mean(axis=1)
            # set lower triangle and diagonal of correlation matrix to NA
            corrmat = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool))
            # where a pairwise correlation is greater than the cutoff value, check whether mean abs.corr of a or b
            # is greater and cut it
            to_delete = list()
            for col in range(0, len(corrmat.columns)):
                for row in range(0, len(corrmat)):
                    if corrmat.iloc[row, col] > self.cor_thresh:
                        # print(f"Compare {corrmat.index.values[row]} with {corrmat.columns.values[col]}")
                        if average_corr.iloc[row] > average_corr.iloc[col]:
                            to_delete.append(corrmat.index.values[row])
                        else:
                            to_delete.append(corrmat.columns.values[col])
            self.sel_columns = list(set(to_delete))
            return self

        else:
            print("Chosen method " + self.method + " does not exist. Defaulted to ignore multicollinearity")
            return self

    def transform(self, X):
        # return the dataframe without the columns that were deleted
        return X.drop(self.sel_columns, axis=1)


# Apply the standardscaler by group (or by the entire column if no group is specified) for a set of specified columns
# from: https://stackoverflow.com/questions/68356000/how-to-standardize-scikit-learn-by-group
class GroupByScaler(BaseEstimator, TransformerMixin):
    def __init__(self, by=None, sel_cols=None):
        self.scalers = dict()
        self.by = by
        self.cols = sel_cols

    def fit(self, X, y=None):
        # make a copy of X to silence setcopy warning
        X = X.copy()
        # if no group was specified, standardize the columns by the entire column (sample)
        if not self.by:
            x_sub = X.loc[:, self.cols]
            self.scalers["no_group"] = StandardScaler().fit(x_sub)
        # if a group was specified, standardize the selected columns by group
        else:
            for val in X.loc[:, self.by].unique():
                mask = X.loc[:, self.by] == val
                X_sub = X.loc[mask, self.cols]
                self.scalers[val] = StandardScaler().fit(X_sub)
        return self

    def transform(self, X, y=None):
        # make a copy of X to silence setcopy warning
        X = X.copy()
        # if no group was specified, standardize the columns by the entire column (sample)
        if not self.by:
            # transform the specified columns with the standardscaler
            X.loc[:, self.cols] = self.scalers["no_group"].transform(X.loc[:, self.cols])
        # if a group was specified, standardize the selected columns by group
        else:
            for val in X.loc[:, self.by].unique():
                mask = X.loc[:, self.by] == val
                X.loc[mask, self.cols] = self.scalers[val].transform(X.loc[mask, self.cols])
        return X


# custom class to remove outliers from the dataset
# there are two outlier removal procedures: trials with task times > 5 minutes are removed, trials are removed
# which have an interquantile based outlier for a specified set of columns
# WARNING: This class removes rows from the dataset (X). It can not be used inside the sk learn pipeline
# the class is highly specified for the current dataset
class RemoveOutliers(BaseEstimator, TransformerMixin):

    def __init__(self, method="cutoff", sel_cols=None, iqr_thresh=2.5, by=None):
        self.iqr = dict()
        self.method = method
        self.sel_cols = sel_cols
        self.iqr_thresh = iqr_thresh
        self.by = by

    def fit(self, X):

        # if the method is the interquantile range method
        if self.method == "iqr":
            # loop every mouse task
            for val in X[self.by].unique():
                # select only the data of the specified mouse task
                mask = X[self.by] == val
                X_sub = X.loc[mask, self.sel_cols]
                # calculate the interquantile outlier removal critical values for each mouse task dataset
                q1 = X_sub.quantile(0.25)
                q3 = X_sub.quantile(0.75)
                iqr = q3 - q1
                # save the values in a dictionary
                self.iqr[val] = {"q1": q1, "q3": q3, "iqr": iqr}

        return self

    def transform(self, X):
        # if the selected method is the cutoff method, just remove trials that have a longer task time than 5 minutes
        if self.method == "cutoff":
            return X.loc[X["task_duration"] < 300]

        # if the specified method is the iqr outlier removal method
        elif self.method == "iqr":
            # prepare a list to store the datasets for each mouse task without outliers
            cleaned_dfs = []
            # loop all
            for val in X[self.by].unique():
                mask = X[self.by] == val
                X_sub = X.loc[mask]
                cleaned_dfs.append(X_sub[~((X_sub[self.sel_cols] < (self.iqr[val]["q1"] - self.iqr_thresh * self.iqr[val]["iqr"]))
                                           | (X_sub[self.sel_cols] > (self.iqr[val]["q3"] + self.iqr_thresh * self.iqr[val]["iqr"]))).any(axis=1)])

            return pd.concat(cleaned_dfs)


#%%

# ----------------
# Dataset Creation
# ----------------

# Create different Training-Test dataset pairs based on the specified preprocessing options
dataset_pairs = {}

# get the outlier removal options that will be looped
outlier_options = [{"method": "cutoff", "thresh": None}, {"method": "iqr", "thresh": 2.5}, {"method": "iqr", "thresh": 3.5}]

# loop the outlier removal options to create the different datasets
for opt in outlier_options:

    # get the option combination
    save_string = opt["method"] + "_" + str(opt["thresh"]) if opt["method"] == "iqr" else opt["method"]
    print(f"Creating the train-test dataset pair for option combination: {save_string}")

    # Notes about the data preprocessing procedure for the mouse task:
    # The data transformation includes the following steps
    # - Removing Outlier datasets (per mouse usage task)
    # - Linear Equation of the mouse usage features to account for potential differences between the  mouse usage tasks
    # - Removal of multicollinear mouse usage features

    # Note that in the mixed-model analysis, we transformed the input features to be normal-like. Such a transformation
    # should not be necessary for the random forest algorithms which will be used to train the data

    # There exist many other potential other transformation procedures. The existing procedure was choosen as a
    # compromise between testing out different options, but not being overwhelmed by too many options
    # Note that the outlier removal procedure simulates the case that new data is not predicted by the machine learning
    # model if it is classified as an outlier, in a real world case, the prediction of a trial that is labelled as an
    # outlier based on the training dataset, will not return a predicted value but a promt that a prediction is not
    # possible for the specified trial
    # There are other options for handling outliers as a preprocessing step:
    # - Remove Outliers based on the entire dataset before splitting into a train-test set: the logic of this option is
    #   that the machine learning analysis should be tested on a "cleaned dataset" that contains no suspicious cases.
    #   This approach does not translate well into a real world application, where outliers will be present
    # - Remove Outliers in the training set only: the rationale is that outliers might appear in the training data
    #   and should be removed for a better prediction model, but predictions are made on all incoming trials (even
    #   if the trial is a potential outlier, a target value for the trial will be predicted)

    # Setup the data preprocessing pipeline to streamline all data transformation
    # The pipeline applies all data transformation processes one after another. It helps to make sure that all
    # transformations are applied correctly and decreases potential errors in the data preprocessing procedure
    dataset_transformation_pipeline = Pipeline([
        # Trans 1: Remove Outliers from the data with the specified removal options
        ("remove_outlier", RemoveOutliers(method=opt["method"], sel_cols=all_predictors, iqr_thresh=opt["thresh"],
                                 by="taskNum")),
        # Trans 2: Standardize (linear equate) by the mouse task (to account for potential differences in the predictors
        # between the different mouse tasks)
        ('equateTask', GroupByScaler(by="taskNum", sel_cols=all_predictors)),
        # Trans 3: Remove collinear features (based on a correlation coefficient threshold of .8
        ('remove_multicoll', HandleMulticollinearity(method="corr", cols=all_predictors, cor_thresh=0.8))
    ])

    # Fit the data transformation pipeline on the train data and transform the train dataset as well as the test dataset
    transformed_train_df = dataset_transformation_pipeline.fit_transform(train_df)
    transformed_test_df = dataset_transformation_pipeline.transform(test_df)

    # save the train-test pair in the train-test-pair dictionary
    dataset_pairs[save_string] = {"train": transformed_train_df, "test": transformed_test_df}


#%%

# Below code is equal to the code of the machine learning analysis of the free mouse usage data

##############################################
# 3. Setup for the Machine Learning Analysis #
##############################################

# --------------------------------------------------------------------------------------------------------
# helper functions to run the machine learning analysis with a given dataset and a given target variable
# --------------------------------------------------------------------------------------------------------


# helper function to plot the feature importance scores of the fitted ml models
def plot_feature_importance_scores(importance_results, col_names, f_name):

    # rename dictionary to give the mouse features "better names" for the plots in the paper
    # this is not a great solution, and is only specific to the plotting, the variable names are saved with their old
    # names in the machine learning pipeline
    rename_dict = {
        # control variables
        'timestamp': 'Timestamp', 'zoom': 'Zoom', 'screen_width': 'Screen Width',
        'screen_height': 'Screen Height', 'order': 'Order', 'age': 'Age', 'sex': 'Sex', 'hand': 'Hand',
        # Mouse Task Features
        "clicks": 'Clicks',
        "task_total_dist": "Task: Tot. Distance",
        "task_angle_sd": 'Task: Angle (sd)',
        "task_x_flips": 'Task: X-Flips',
        "task_y_flips": 'Task: Y-Flips',
        "trial_sd_duration": 'Trial (sd): Duration',
        "trial_mean_trial_move_offset": 'Trial (mean): Initiation Time',
        "trial_sd_trial_move_offset": 'Trial (sd): Initiation Time',
        "trial_sd_total_dist": 'Trial (sd): Tot. Distance',
        "trial_sd_distance_overshoot": 'Trial (sd): Ideal Line Deviation',
        "trial_mean_speed_mean": 'Trial (mean): Speed (mean)',
        "trial_sd_speed_mean": 'Trial (sd): Speed (mean)',
        "trial_mean_speed_sd": 'Trial (mean): Speed (sd)',
        "trial_sd_speed_sd": 'Trial (sd): Speed (sd)',
        "trial_sd_abs_jerk_sd": 'Trial (sd): Jerk (sd)',
        "trial_mean_angle_mean": 'Trial (mean): Angle (mean)',
        "trial_sd_angle_mean": 'Trial (sd): Angle (mean)',
        "trial_mean_angle_sd": 'Trial (mean): Angle (sd)',
        "trial_sd_angle_sd": 'Trial (sd): Angle (sd)',
        "trial_sd_x_flips": 'Trial (sd): X-Flips',
        "trial_sd_y_flips": 'Trial (sd): Y-Flips',
        "trial_mean_duration": 'Trial (mean): Duration',
        "trial_mean_total_dist": 'Trial (mean): Tot. Distance',
        "trial_sd_abs_jerk_mean": 'Trial (sd): Jerk (mean)',
        "task_duration": 'Task: Duration',
        "task_abs_jerk_sd": 'Task: Jerk (sd)',
        "trial_mean_abs_jerk_mean": 'Trial (mean): Jerk (mean)',
    }

    # Get the Importance results scores and convert them to a dictionary with each column representing the feature
    # importance scores for one predictor in the machine learning model
    importance_df = pd.DataFrame(importance_results["importances"]).T
    importance_df.columns = [rename_dict[name] if name in rename_dict else name for name in col_names]
    # sort the dataframe by the mean importance score
    importance_df = importance_df.reindex(importance_df.mean().sort_values().index, axis=1)

    # create a barplot figure to visualize the importance scores
    plt.figure(figsize=(22, 16))
    sns.set(font_scale=2.4)
    sns.set_style("whitegrid")
    g = sns.barplot(data=importance_df, orient="h", palette="deep", errwidth=6)
    g.set(xlim=(0, importance_df.to_numpy().max()))
    plt.tight_layout()

    # save the figure
    # plt.savefig(f_name + "_feat_import.png")

    plt.show()


# helper function for the machine learning regression analysis
# to give more control about the results, it would also be possible to return the trained model, save it, and be
# able to get results in a later step without having to wait for the training process over and over again
def ml_regression(training_data, test_data, predictors, target):

    # split the data into the predictors and targets
    x_train, x_test = training_data.loc[:, predictors], test_data.loc[:, predictors]
    y_train, y_test = training_data[target], test_data[target]

    print(f"Shape of the train-test datasets: {x_train.shape, x_test.shape, y_train.shape, y_test.shape}")

    # The machine learning procedure has three parts:

    # In the first part, model hyperparameters are selected using Randomized Hyperparameter Cross Validation Search
    # see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

    # In the second part, the tuned model is fit on the training dataset and the predicton performance is tested using
    # the test dataset

    # In the third part, the machine learning model is interpreted -> the importance of the input features for the
    # model prediction are evaluated

    # Hyperparameter selection:
    # -------------------------

    # setup the random forest regressor
    # we use the random forest, because it is a common ML algorithm that has been successfully used in the context of
    # machine learning analysis with mouse usage data
    # Advantage of the random forest regression:
    # - does not care if values are on different scales (this is less important, because we standardized the features)
    # - does not predict outside of the feature range (value between 0-100)
    rf_regressor = RandomForestRegressor()

    # setup a grid of hyperparameters that will be tuned in RandomizedSearchCV
    rf_hyperparameters = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': [1.0, 'sqrt', "log2"],
        'max_depth': [60, 70, 80, 90, 100, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # setup Cross Validation
    # 5-fold cross validation is used. Participants are stratified across the groups in order to balance
    # out the number of participants during the cross validation process
    # setup the CV generator
    cv_generator = StratifiedKFold(n_splits=5)
    # create the cross validation splits, which are stratified by the participant -> "ID" column of the dataset
    # The X-Input is a placeholder, because it is not required for splitting:
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold.split
    cv_splits = [(train, test) for train, test in cv_generator.split(
        X=np.zeros(len(training_data)),
        y=training_data["ID"])]

    # run the RandomizedSearchCv with the selected random forest regressor, the random forest hyperparameter grid, and
    # the cross validation splits, select the r2_score as the score for selecting the best hyperparameter pairs,
    # use multiple cores (n_jobs), refit the entire model after having selected the best parameters and specify
    # a random state
    print("Running the Randomized Search Cross Validation")
    ml_model = RandomizedSearchCV(rf_regressor, rf_hyperparameters, cv=cv_splits, scoring=make_scorer(r2_score),
                                  refit=True, n_jobs=None, random_state=rng)

    # Evaluating the tuned model
    # --------------------------

    # fit the training dataset to the tuned model
    ml_model.fit(x_train, y_train)

    # get the selected hyperparameters
    selected_hyperparameters = ml_model.best_params_
    print(f"Selected Hyperparameters: {selected_hyperparameters}")
    # It is also possible to get additional infos from the RandomizedSearchCV procedure
    # print(testing.cv_results_)
    # print(testing.best_score_)

    # make predictions on the test set
    predictions = ml_model.predict(x_test)

    # get the mean absolute error and R²-score as the regression evaluation metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"R²-Prediction Score: {r2}")
    print(f"Mean Absolute Prediction Error: {mae}")
    print(f"Mean Squared Prediction Error: {mse}")


    # Get the feature importance scores to "interpret" the machine learning model
    # ---------------------------------------------------------------------------
    print(f"Running the feature importance permutation")
    # use feature importance permutation
    # see: https://scikit-learn.org/stable/modules/permutation_importance.html
    feature_importance_scores = permutation_importance(ml_model, x_test, y_test,
                                                       scoring=make_scorer(r2_score),
                                                       n_jobs=None,
                                                       n_repeats=30,
                                                       random_state=rng)

    print(f"Analysis are done")
    # return the results of the regression analysis
    results = {"hyperparams": selected_hyperparameters, "scores": {"r2": r2, "mae": mae, "mse": mse},
               "feat_importance": feature_importance_scores}

    return results


#%%

# All Below code is equal to the code of the machine learning analysis of the free mouse usage (except for
# name of the result file that is generated)

########################################
# 4. Run the Machine Learning Analysis #
########################################

# --------------------------------------------------------------------------------------
# Playground to run the machine learning analysis for a single dataset & target variable
# --------------------------------------------------------------------------------------

# Draw a random dataset and a random target variable
playground_dset_name, playground_dset = random.choice(list(dataset_pairs.items()))
playground_target = random.choice(all_targets)
# get the predictors for the randomly drawn dataset
playground_preds = [pred for pred in list(playground_dset["train"].columns) if pred in all_predictors]

#%%

# run the machine learning model with the specifications of the playground data
print(f"Run the machine learning analysis with dataset {playground_dset_name} and target: {playground_target}")
playground_results = ml_regression(playground_dset["train"], playground_dset["test"], playground_preds, playground_target)



#%%

# Play around with the results (some information was already shown in the machine learning function):

# go through all results
for i in playground_results["scores"]:
    print(f"Score: {i}")
    print(playground_results["scores"][i])

# show the selected hyperparameters
print(f"Selected Hyperparameters:\n{playground_results['hyperparams']}")

# show the mean feature importance scores
print("\nMean Feature Importance Scores")
print(playground_results["feat_importance"]["importances_mean"])

# visualize the importance scores
plot_feature_importance_scores(playground_results["feat_importance"], playground_preds, "playground")


#%%

# -----------------------------------------------------------------------------------------
# Run the analysis for all dataset - target iterations and save the results in a dictionary
# -----------------------------------------------------------------------------------------

ml_analysis_results = {}

# loop all targets
for target in all_targets:
    # loop all dataset pairs
    for dset in dataset_pairs:
        # for each target-dataset combination, we calculate two models:
        # 1. A "baseline model" that includes only the participant ID (this model is similar to the random intercept
        #       only model in the mixed model analysis)
        # 2. A "full model" that additionally includes all mouse-usage features
        # The baseline model is calculated to be able to get the "increment" in prediction performance by the
        # mouse usage features

        # create a dic to store the results of the specific target-dset combination
        ml_analysis_results[target + "_" + dset] = {}

        # get the predictors of the baseline model
        baseline_preds = ["ID"]
        # get only the mouse usage predictors (not all are relevant because some got removed due to collinearity)
        mouse_preds = [pred for pred in list(dataset_pairs[dset]['train'].columns)
                                             if pred in all_predictors]

        print(f"Running the machine learning analysis for the BASELINE MODEL using dataset: {dset} and "
              f"target: {target}")
              
        baseline_results = ml_regression(dataset_pairs[dset]["train"], dataset_pairs[dset]["test"],
                                                 baseline_preds, target)

        print(f"Running the machine learning analysis for the FULL MODEL using dataset: {dset} and "
              f"target: {target}")
        full_model_results = ml_regression(dataset_pairs[dset]["train"], dataset_pairs[dset]["test"],
                                               baseline_preds + mouse_preds, target)

        # save the results + information about the train/test dataset shapes & the mouse usage predictors
        ml_analysis_results[target + "_" + dset]["baseline_results"] = baseline_results
        ml_analysis_results[target + "_" + dset]["full_model_results"] = full_model_results
        ml_analysis_results[target + "_" + dset]["dset_shapes"] = {"train_shape": dataset_pairs[dset]["train"].shape,
                                                                   "test_shape": dataset_pairs[dset]["test"].shape}
        ml_analysis_results[target + "_" + dset]["predictors"] = {"baseline": baseline_preds,
                                                                  "mouse_only": baseline_preds + mouse_preds}

#%%

# save the results as a pickle file
with open("Mouse_task_ML_Results.p", 'wb') as fp:
    pickle.dump(ml_analysis_results, fp, protocol=pickle.HIGHEST_PROTOCOL)


#%%

# save the results as a json file

# the json file does not accept numpy arrays, so they need to be converted first
# copied from: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


json_dumbed = json.dumps(ml_analysis_results, cls=NumpyEncoder)

with open("mouse_task_ML_results.json", 'w') as f:
    json.dump(json_dumbed, f)


#%%

# if already calculated, the results can be imported here:
with open('Mouse_Task_Results/Machine_Learning/mouse_task_ML_results.p', 'rb') as handle:
    ml_analysis_results = pickle.load(handle)

#%%

# ----------------
# View the results
# ----------------

# simple step by step viewer of the results (shows and saves the permutation + confusion matrix plots)
# if results are already calculated and saved, replace the ml_analysis_results with loaded_results
for i in ml_analysis_results:
    print(f"Printing Results for: {i}")
    if input("Print the results? [y/n]") == "y":
        # print baseline results & show the feature importance plots & confusion matrix, if the target is stress
        print("Baseline Results:")
        print(ml_analysis_results[i]["baseline_results"]["scores"])
        plot_feature_importance_scores(ml_analysis_results[i]["baseline_results"]["feat_importance"],
                                       ml_analysis_results[i]["predictors"]["baseline"], 'baseline_' + i)
        # print full model results and show the feature importance plots confusion matrix, if the target is stress
        print("Full Model Results:")
        print(ml_analysis_results[i]["full_model_results"]["scores"])
        plot_feature_importance_scores(ml_analysis_results[i]["full_model_results"]["feat_importance"],
                                       ml_analysis_results[i]["predictors"]["full_model"], 'full_mod_' + i)
        print("\n")
    else:
        print("Stopped manually printing the results")
        break


#%%

# run additional test analysis to look at the effect of a potential inclusion of the control variables

# To do so, randomly draw a dataset and target
controls_test_dset_name, controls_test_dset = random.choice(list(dataset_pairs.items()))
controls_test_target = random.choice(all_targets)
# get the predictors for the randomly drawn dataset
controls_test_mouse_features = [pred for pred in list(controls_test_dset["train"].columns) if pred in all_predictors]

#%%

# first, run a model that only includes the control variables (+ participant ID)
controls_only_model = ml_regression(controls_test_dset["train"], controls_test_dset["test"],
                                    ['ID'] + covariates, controls_test_target)

#%%

# second, run a model that additionally includes all mouse usage features
controls_and_mouse_model = ml_regression(controls_test_dset["train"], controls_test_dset["test"],
                                         ['ID'] + covariates + controls_test_mouse_features, controls_test_target)

#%%

# Run one more model with changes to the target variable (instead of the absolute level, use the change score of the
# target from the mean of the participant)

# Function to do a cluster mean centering of a target variable (function should probably be improved)
def clusterMeanCenter(train_data, test_data, target):

    # first, calculate the group mean in the cluster of the training dataset
    train_data[target + "_clusterMean"] = train_data.groupby(["ID"])[target].transform('mean')
    # cluster mean center the target variable
    train_data[target + '_CMC'] = train_data[target + "_clusterMean"] - train_data[target]

    # now add the cluster mean to the test dataset
    test_data = test_data.merge(train_data.drop_duplicates(subset='ID', keep="last").loc[:, ["ID", target + "_clusterMean"]],
                        on='ID', how='right')
    # calculate the clustered mean center target variable in the test dataset
    test_data[target + '_CMC'] = test_data[target] - test_data[target + "_clusterMean"]

    # return the training and test dataset
    return train_data, test_data

# draw a random dataset again
center_test_dset_name, center_test_dset = random.choice(list(dataset_pairs.items()))
center_test_target = random.choice(all_targets)
# get the predictors for the randomly drawn dataset
center_test_mouse_features = [pred for pred in list(center_test_dset["train"].columns) if pred in all_predictors]

# get the training and test dataset with cluster mean centered target
center_test_train, center_test_test = clusterMeanCenter(center_test_dset["train"], center_test_dset["test"],
                                                        center_test_target)

#%%

# run the machine learning model
ml_regression(center_test_train, center_test_test, ['ID'] + center_test_mouse_features, center_test_target + "_CMC")
