'''
Code to reanalyze the data of Banholzer et al. (2021) with machine learning in order to test the prediction
performance of their proposed speed-accuracy model
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, classification_report, \
    confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# Create custom classes for data transformation in the sklearn pipeline
from sklearn.base import TransformerMixin, BaseEstimator

#%%

# dataset import
replication_data = pd.read_csv('Data_Banholzer_et_al.csv')

# transform the ID column from strings to labels
replication_data["user"] = LabelEncoder().fit_transform(replication_data["user"])

#%%

# replicate the data preprocessing procdure to get the 'original replication dataset'

# first, exclude rows with na speed values
data = replication_data.dropna(subset=["speed"])
# next, exclude all samples with less than 10 complete mouse trajectories
data = data.loc[data["n_traj"] >= 10]
# we also need to drop participants with only 1 dataset from the analysis, because a single datapoint can not be used
# for training and testing
data = data[data.groupby('user').user.transform('count') > 1]

#%%

# count the users and the number of their dataframes
data.user.value_counts()

#%%
# now setup the machine learning pipeline

# this included splitting the data into a training and test set as well as standardizing the data (which will be done
# in the training dataset and then applied to the test dataset. Furthermore, Banholzer et al. (2021) first standardized
# their data and calculated the speed-accuracy tradeoff interaction afterwards

# the data has no information about the order of the collected data per participant, so we can not split the data
# by order to include the first 80% of the data as a training dataset and the remaining (last) 20% as a test dataset
# instead we randoml draw 80% of the participant data and use it for training and use the remaining 20% for testing

# outsource it into a function in order to be able to repeat this process to getter a better feeling for the results


def get_train_test_data(data):

    # first, split the entire dataset in data subsets for each participant
    split_dataframes = [y for x, y in data.groupby("user", as_index=True)]

    train_dfs = []
    test_dfs = []
    # iterate the dataframe of each participant
    # measurements
    for df in split_dataframes:
        # shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
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

    return train_df, test_df


#%%

# pipeline helper functions

# A simple class to calculate the speed-accuracy interaction variable
class CalculateSpeedAccInteraction(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    # only requires a transform
    def transform(self, X):
        # calculate the interaction
        X['interaction'] = X['speed'] * X['accuracy']
        return X


# Apply the standardscaler by group (or by the entire column if no group is specified) for a set of specified columns
# from: https://stackoverflow.com/questions/68356000/how-to-standardize-scikit-learn-by-group
class StandardizeByColumn(BaseEstimator, TransformerMixin):
    def __init__(self, sel_cols=None):
        self.scalers = dict()
        self.cols = sel_cols

    def fit(self, X, y=None):
        # make a copy of X to silence setcopy warning
        X = X.copy()
        #standardize the selected columns
        x_sub = X.loc[:, self.cols]
        self.scalers["std"] = StandardScaler().fit(x_sub)

        return self

    def transform(self, X, y=None):
        # make a copy of X to silence setcopy warning
        X = X.copy()
        # transform the specified columns with the standardscaler
        X.loc[:, self.cols] = self.scalers["std"].transform(X.loc[:, self.cols])

        return X

#%%

# helper function for the machine learning classification analysis
def ml_classification(training_data, test_data, predictors, target):

    # split the data into the predictors and targets
    x_train, x_test = training_data.loc[:, predictors], test_data.loc[:, predictors]
    y_train, y_test = training_data[target], test_data[target]

    # standardize the data and calculate the speed-accuracy interaction

    print(f"Shape of the train-test datasets: {x_train.shape, x_test.shape, y_train.shape, y_test.shape}")

    # Model Training, Testing and Evaluation
    # -------------------------

    # setup the random forest classifier
    # we used the classifier with the default hyperparameter settings. Tuning the hyperparameters usually improves the
    # fit only marginally and the default setting are "good" enough to detect if it is possible to classify
    # additionally, we are more interested in the comparison between a fit with the id only and a fit with the
    # id + the mouse usage parameters
    rf_classifier = RandomForestClassifier()

    # fit the training dataset to the classifier
    rf_classifier.fit(x_train, y_train)

    # make predictions on the test set
    predictions = rf_classifier.predict(x_test)

    # get classification model scores
    cm = confusion_matrix(y_test, predictions)
    cr = classification_report(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    b_acc = balanced_accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    # print some metrics
    print(f"Balanced Accuracy Score: {b_acc}")
    print(f"F1_Score: {f1}")
    print(f"Accuracy Score: {acc}")

    # return the results of the regression analysis
    results = {"scores": {"cm": cm, "cr": cr, "acc": acc, "b_acc": b_acc, "f1": f1}}

    return results


# helper function for the machine learning regression analysis
def ml_regression(training_data, test_data, predictors, target):

    # split the data into the predictors and targets
    x_train, x_test = training_data.loc[:, predictors], test_data.loc[:, predictors]
    y_train, y_test = training_data[target], test_data[target]

    print(f"Shape of the train-test datasets: {x_train.shape, x_test.shape, y_train.shape, y_test.shape}")

    # setup the random forest regressor
    rf_regressor = RandomForestRegressor()

    # fit the training dataset to the tuned model
    rf_regressor.fit(x_train, y_train)

    # make predictions on the test set
    predictions = rf_regressor.predict(x_test)

    # get the mean absolute error and R²-score as the regression evaluation metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"R²-Prediction Score: {r2}")
    print(f"Mean Absolute Prediction Error: {mae}")

    # return the results of the regression analysis
    results = {"scores": {"r2": r2, "mae": mae}}

    return results

#%%


# run the classification
classification_results = {}

# repeat it 5 times to get a better feeling for the results
for i in range(5):
    classification_results[i] = {}

    # create the randomized dataset
    training_data, test_data = get_train_test_data(data)

    # setup the transformation pipeline
    pipe = Pipeline([
        # standardize
        ('standardize', StandardizeByColumn(sel_cols=['accuracy', 'speed'])),
        # calculate interaction
        ('calc_interaction', CalculateSpeedAccInteraction())
    ])

    transformed_train = pipe.fit_transform(training_data)
    print(transformed_train["stress"].value_counts())
    transformed_test = pipe.transform(test_data)
    print(transformed_test["stress"].value_counts())

    # run the machine learning classification for the baseline model that only uses the user id as an input
    print("Run the Baseline Model Classification")
    classification_results[i]["Baseline"] = ml_classification(transformed_train, transformed_test,
                                                  ['user'], 'stress')

    print("Run the Mouse Model Classification")
    # run the machine learning classification with the user id and the mouse usage features
    classification_results[i]["Mouse"] = ml_classification(transformed_train, transformed_test,
                                                  ['speed', 'accuracy', 'interaction', 'user'], 'stress')


#%%

# run the regression on valence and arousal

# run the classification
regression_results = {}

for target in ["arousal", "valence"]:
    print(f'\nRunning the analysis with target: {target}')
    regression_results[target] = {}
    # repeat it 5 times to get a better feeling for the results
    for i in range(5):
        regression_results[target][i] = {}

        # create the randomized dataset
        training_data, test_data = get_train_test_data(data)

        # setup the transformation pipeline
        pipe = Pipeline([
            # standardize
            ('standardize', StandardizeByColumn(sel_cols=['accuracy', 'speed'])),
            # calculate interaction
            ('calc_interaction', CalculateSpeedAccInteraction())
        ])

        transformed_train = pipe.fit_transform(training_data)
        transformed_test = pipe.transform(test_data)

        # run the machine learning classification for the baseline model that only uses the user id as an input
        print("Run the Baseline Model Regression")
        regression_results[target][i]["Baseline"] = ml_regression(transformed_train, transformed_test,
                                                      ['user'], target)

        print("Run the Mouse Model Regression")
        # run the machine learning classification with the user id and the mouse usage features
        regression_results[target][i]["Mouse"] = ml_regression(transformed_train, transformed_test,
                                                      ['speed', 'accuracy', 'interaction', 'user'], target)
        print("\n")


#%%

# save all results in one dictionary

ml_results = {"Regression": regression_results, "Classification": classification_results}

# save the results as a pickle file
with open("Banholzer_ML_Analysis_results.p", 'wb') as fp:
    pickle.dump(ml_results, fp, protocol=pickle.HIGHEST_PROTOCOL)


#%%

# if the results have already been saved, they can also be loaded here
# if already calculated, the results can be imported here:
with open('Banholzer_ML_Analysis_results.p', 'rb') as handle:
    ml_results = pickle.load(handle)

classification_results = ml_results["Classification"]
regression_results = ml_results["Regression"]

#%%

# get stats about the classification results
# loop the repeated arousal result iterations
class_result_dict = {
        "Baseline": {"acc": [], "b_acc": [], "f1": []},
        "Mouse": {"acc": [], "b_acc": [], "f1": []}
    }

for iteration in classification_results:
    for model in classification_results[iteration]:
        for score in classification_results[iteration][model]["scores"]:
            if score != "cr" and score != "cm":
                class_result_dict[model][score].append(classification_results[iteration][model]["scores"][score])

for mod in class_result_dict:
    for score in class_result_dict[mod]:
        print(f"{score} in {mod}: Mean = {np.mean(class_result_dict[mod][score])}, SD = {np.std(class_result_dict[mod][score])}")

#%%

# get stats about the regression results
def print_regression_performance(result_data):
    # loop the repeated arousal result iterations
    result_dict = {
        "Baseline": {"r2": [], "mae": []},
        "Mouse": {"r2": [], "mae": []},
    }

    for iteration in result_data:
        for model in result_data[iteration]:
            for score in result_data[iteration][model]["scores"]:
                result_dict[model][score].append(result_data[iteration][model]["scores"][score])

    for mod in result_dict:
        for score in result_dict[mod]:
            print(f"{score} in {mod}: Mean = {np.mean(result_dict[mod][score])}, SD = {np.std(result_dict[mod][score])}")


# valence
print("Valence Results")
print_regression_performance(regression_results['valence'])

# arousal
print("Arousal results")
print_regression_performance(regression_results['arousal'])