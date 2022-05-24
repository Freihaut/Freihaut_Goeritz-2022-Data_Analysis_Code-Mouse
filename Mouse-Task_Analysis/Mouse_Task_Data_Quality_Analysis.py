'''
Code to evaluate the data quality of the tracked mouse task data in order to be able to make educated guesses
about data removal procedures/rules of potential outliers/bad cases. The data quality is inspected with rules-of-thumbs.
The mouse movement data is visually inspected for selected potential bad cases (univariate outliers of selected
variables)
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''

#%%

# library imports
import numpy as np
import pandas as pd
import json
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

#%%

# import the raw dataset
with gzip.open("Datasets_Raw/MouseTask_dataset.json.gz", "rb") as f:
    raw_data = json.loads(f.read())

# import the processed mouse task features
data_quality_features = pd.read_csv("Mouse_Task_Features_Data_Quality_Inspection.csv")


#%%

# calculate a new potential data quality feature: the relationship between the number of movement datapoints and
# the median logged sample frequency as a potential indicator of unusual movement patterns
# relative movement datapoints
data_quality_features["rel_mo_dpoints"] = data_quality_features["median_sampling_freq"] * data_quality_features["move_datapoints"] / 100


#%%

# mouse data cleaning helper function (the same as in the task feature calculation procedure)
def clean_mouse_task_data(mouse_data):

    # sort the dictionary by the timestamp to get the recorded mouse datapoints chronologically
    sorted_page_data = sorted(mouse_data.items(), key=lambda x: x[1]["t"])

    # remove all data points from the task dataset that were made before the "start of the task" = click on the first
    # circle in the point-and-click task
    sorted_page_data = [i for i in sorted_page_data if 7 > i[1]["cN"] > 0]

    # set variables for the mouse events Movement and Click
    key, value1, value2 = "e", "Mo", "Cl"

    # Save the last datapoint
    last_coordinates = [0, 0]
    last_timestamp = 0

    # save the last clicktime (for cleaning potential click-artifacts)
    clicktimes = []

    # save the cleaned datapoints
    clean_list = []

    # count the number of removed datapoints and total datapoints
    artifacts = 0
    total_movement_points = 0
    valid_movement_datapoints = 0

    # Loop over all datatuples of the sorted_page_data dictionary
    for data_tuple in sorted_page_data:
        # the second touple is the datapoint, the first tuple the dictionary key
        datapoint = data_tuple[1]
        # if its a mousePositionChanged Datapoint
        if datapoint[key] == value1:
            # and if the x- & y- coordinates are not equal to the previous datapoint or the timestamps are not equal
            if ([datapoint["x"], datapoint["y"]] != last_coordinates) and (datapoint["t"] > last_timestamp):
                # save the datapoint in the clean list
                clean_list.append(datapoint)
                # save the coordinates and the timestamp of the datapoint
                last_coordinates = [datapoint["x"], datapoint["y"]]
                last_timestamp = datapoint["t"]
                # increase the movement data point counter
                valid_movement_datapoints += 1
            else:
                # increase the artifact counter
                # print(last_coordinates, last_timestamp, [datapoint["x"], datapoint["y"]], datapoint["time"])
                artifacts += 1
            # increase the datapoint counter
            total_movement_points += 1
        elif datapoint[key] == value2:
            # if its a mouse click datapoint and the timestamp has not previously been in the dataset (= artifact)
            if datapoint["t"] not in clicktimes:
                clean_list.append(datapoint)
                clicktimes.append(datapoint["t"])

    # return the cleaned mouse data list and the percentage of artifacts
    return clean_list


#%%


# Helper Function to visualize the mouse movement
def visualize_mouse_movement(dframe, title):

    # select only the movement data
    dframe = dframe.loc[dframe["e"] == "Mo"]

    # simple scatterplot of the raw mouse usage data with connected consecutive mouse movement datapoints. The
    # data has unique colors per trial
    colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'}

    plt.scatter(x=dframe["x"], y=dframe["y"], color=dframe["cN"].map(colors), marker=".")
    plt.plot(dframe["x"], dframe["y"], linestyle="--", color="black", linewidth=1.25)
    grouped = dframe.groupby("cN")
    for key, group in grouped:
        plt.plot(group["x"], group["y"], color=colors[key])

    # set a title
    plt.title(title)
    # hide the axis
    plt.axis("off")
    # show the plot
    plt.show()


#%%

# helper function to sort the feature dataset by columns and assess the data quality by inspecting the visualized data
def manually_inspect_data_quality(dframe, target_variable, ascend=True):

    # sort the dataframe by the target column in the specified order
    sorted_df = dframe.sort_values(by=[target_variable], ascending=ascend)

    # loop over the rows of the sorted dataframe to be able to select the correct raw data for visualization
    # note that looping dataframe rows usually is an anti-pattern and should be avoided
    for index, row in sorted_df.iterrows():
        if input("Do you want to inspect the next case? [y/n]") == "y":

            # get the raw data of the participant and clean it
            par_raw_data = raw_data[row["ID"]]["TaskData"][row["dset_ID"]]["mTaskData"]
            clean_data = clean_mouse_task_data(par_raw_data)
            # visualize the cleaned raw data
            print(f"Dset Value for {target_variable}: {row[target_variable]} for {row['ID']} and {row['dset_ID']}")
            visualize_mouse_movement(pd.DataFrame(clean_data), target_variable)
        else:
            print("Stopped the manual data inspection")
            break


#%%

##############################################################################
# Visually Inspect potentially bad datasets based on selected mouse features #
##############################################################################

# number of available movement datapoints, ascending
manually_inspect_data_quality(data_quality_features, "move_datapoints", ascend=True)

# Inspection Results: #
# There are some cases with very little movement datapoints, which must be removed from the dataset, because mouse
# movement was not logged correctly. After a jump from 10 datapoints to 88 datapoints, the recorded movement data looks
# normal: Datasets with 10 or less movement datapoints should be removed from the dataset


#%%

# median logging interval, descending
manually_inspect_data_quality(data_quality_features, "median_sampling_freq", ascend=False)

# Inspection Results: #
# Similar to the number of movement datapoints, a very high logging interval indicates that the data was not recorded
# correctly. After the jump from 144 to 46, the data looks normal: datasets with a logging interval of 144 or higher
# should be removed

#%%

# relative number of movement datapoints in respect to the logging frequency, descending (how do cases with "many"
# datapoints look like? Filtering in ascending order should not add any information to inspecting by the number
# of datapoints and the median sampling frequency)
manually_inspect_data_quality(data_quality_features, "rel_mo_dpoints", ascend=False)

# Inspection Results: #
# There is no cut-off that indicates that participants did not do the task correctly. In some cases, participants
# likely moved the mouse around a bit, in other cases participants likely moved rather slow and therefore produced many
# datapoints. This variable should not be used as an indicator of bad data quality

#%%

# artifact percentage
manually_inspect_data_quality(data_quality_features, "artifact_percent", ascend=False)
# Inspection Results: #
# cases with high amount of artifacts (which get removed in the data cleaning procedure) do not show signs of potential
# data quality problems


#%%

# Task Duration (ascending)
manually_inspect_data_quality(data_quality_features, "task_duration", ascend=True)
# Inspection Results: #
# Very fast task duration times do not show signs of recording problems or other data quality issues


#%%

# Task Duration (descending)
manually_inspect_data_quality(data_quality_features, "task_duration", ascend=False)
# Inspection Results: #
# Long task duration is only partionally indicated by the movement (long task execution times should still be removed
# because the task was not done as intended)

#%%

# distance relative to the ideal line, descending
manually_inspect_data_quality(data_quality_features, "trial_mean_distance_overshoot", ascend=False)
# Inspection Results: #
# A high mean overshoot indicates that the task (or some trials) were not done as intented (the mouse was not
# directly moved to the target, but wandered around. There is no obvious cut-off, which separetes "good data" from
# "bad data"


#%%

# total distance, descending
manually_inspect_data_quality(data_quality_features, "task_total_dist", ascend=False)
# Inspection Results: #
# Similar to distance overshoot, there are participants who moved the mouse around/not on a straight line between the
# targets (distance and overshoot do not necessarily mark the same datasets, note that the task distance naturally
# differs between different mouse tasks


#%%

# average angle
manually_inspect_data_quality(data_quality_features, "task_angle_mean", ascend=True)
# Inspection Results: #
# No noticable patterns of problematic data quality with very high or low angle values

#%%

# average speed, fast to slow
manually_inspect_data_quality(data_quality_features, "task_mean_speed", ascend=False)
# Inspection Results: #
# No noticable "problems" with very fast average speed values


#%%

# Inspect the task num raw data to create the sample image for participant mouse usage behavior in task 16
# (the final image in the paper is flipped and the image size is set to figure(figsize=(6, 6)))
manually_inspect_data_quality(data_quality_features.loc[data_quality_features["taskNum"] == 16], "taskNum", ascend=True)


#%%

# Other features could also be tested out
########################################################
# Careful conclusion of the visual data quality analysis
#########################################################

# The most obvious data quality issues are visible with datasets that only include very little recorded datapoints/a
# high median logging interval. Out of the calculated mouse usage features, which describe the mouse usage behavior
# during the task, the most promising feature for data quality analysis based on visual data inspection is the distance
# related feature --> high distance values indicate potential faulty participant behavior (e.g. the participant
# moved the mouse around without following the task instructions)
# Additionally, trials with long task time should be removed (but are less "obvious" with the visual data quality
# inspection). If participants take too long, they likely took a break and timing related features are affected
# by long task times.


#%%

# Compare Outlier Removal Procedures with the selected Variables "task duration" and "mouse distance" based on the
# visual inspection of potential bad data quality

# first, remove cases with obivous recording errors
cleaned_features = data_quality_features.loc[(data_quality_features["median_sampling_freq"] <= 46) &
                                                (data_quality_features["move_datapoints"] >= 88)]

print(cleaned_features.shape)

#%%

# second, inspect the task duration and mouse distance using basic plots
sns.displot(data=cleaned_features, x="task_duration")
plt.show()
sns.boxplot(x=cleaned_features["task_duration"])
plt.show()

# It takes long to visualize the displot (= histogram). The extreme outlier vaule does not allow proper visualization

#%%

# same for distance measure
sns.displot(data=cleaned_features, x="trial_mean_distance_overshoot")
plt.show()
sns.boxplot(x=cleaned_features["trial_mean_distance_overshoot"])
plt.show()

# The visualization indicates that there are a number of outliers with a too big distance overshoot, which is an indicator
# of bad data quality, i.e. participants did not do the task as intended

#%%

# Compare z-outlier removal procedure with an IQR-outlier removal procedure


# helper functions to get outliers based on z-values
def remove_z(dframe, column, thresh=3.0):

    # outliers = dframe.loc[~(np.abs(dframe[column]-dframe[column].mean()) <= (thresh*dframe[column].std()))]

    outliers = dframe.loc[np.abs((dframe[column] - dframe[column].mean()) / dframe[column].std()) > thresh]

    print(f"z-outliers for column {column}: {len(outliers)}")

    return outliers


# helper function to get outliers based on the IQR
def remove_iqr(dframe, column, thresh=3.0):

    q1 = cleaned_features[column].quantile(0.25)
    q3 = cleaned_features[column].quantile(0.75)
    iqr = q3 - q1

    outliers = dframe.loc[(dframe[column] < q1 - thresh * iqr) | (dframe[column] > q3 + thresh * iqr)]

    print(f"iqr-outliers for column {column}: {len(outliers)}")

    return outliers

#%%


# compare the outlier removal procedures for task duration and distance overshoot
z_duration_outlier = remove_z(cleaned_features, "task_duration", thresh=3.0)
iqr_duration_outlier = remove_iqr(cleaned_features, "task_duration", thresh=3.0)

z_distance_outlier = remove_z(cleaned_features, "trial_mean_distance_overshoot", thresh=3.0)
iqr_distance_outlier = remove_iqr(cleaned_features, "trial_mean_distance_overshoot", thresh=3.0)

# The outlier removal based on z-values does not work well with task duration because of an extremly high task
# duration value (-> IQR should be used as a more robust univariate outlier removal procedure)

# get the datframes without outliers
iqr_duration_cleaned = cleaned_features.drop(iqr_duration_outlier.index)
z_duration_cleaned = cleaned_features.drop(z_duration_outlier.index)
print(f"Max: Task Duration After Outlier Removal: {iqr_duration_cleaned['task_duration'].max()}")

iqr_dist_cleaned = cleaned_features.drop(iqr_distance_outlier.index)
z_dist_cleaned = cleaned_features.drop(z_distance_outlier.index)

#%%

# Visual Data Inspection After Outlier Removal
manually_inspect_data_quality(iqr_dist_cleaned, "trial_mean_distance_overshoot", ascend=False)
# Inspection Results: #
# The first 5-10 cases do not show any obvious sign of "bad mouse usage behavior"

#%%

# additionally inspect the "least outlier cases" of the outliers
manually_inspect_data_quality(iqr_distance_outlier, "trial_mean_distance_overshoot", ascend=True)
# Inspection Results: #
# The removed data entries do not all look like "bad data", the removal procedure likely also removes false positives


#%%

# get descriptive stats about the target variable for the removed outliers and cleaned data
# & visualize potential differences in valence & arousal between the outlier and non-outlier group
print(f"Outlier Mean Arousal: {iqr_distance_outlier['arousal'].mean()}, SD: {iqr_distance_outlier['arousal'].std()}")
print(f"Cleaned Data Mean Arousal: {iqr_dist_cleaned['arousal'].mean()}, SD: {iqr_dist_cleaned['arousal'].std()}")
fig, (ax1, ax2) = plt.subplots(2, 1)
sns.boxplot(data=iqr_distance_outlier, x ='arousal', color="blue", ax=ax1)
ax1.set(xlabel=None)
sns.boxplot(data=iqr_dist_cleaned, x='arousal', color="red", ax=ax2)
plt.show()

print(f"Outlier Mean valence: {iqr_distance_outlier['valence'].mean()}, SD: {iqr_distance_outlier['valence'].std()}")
print(f"Cleaned Data Mean valence: {iqr_dist_cleaned['valence'].mean()}, SD: {iqr_dist_cleaned['valence'].std()}")
fig, (ax1, ax2) = plt.subplots(2, 1)
sns.boxplot(data=iqr_distance_outlier, x ='valence', color="blue", ax=ax1)
ax1.set(xlabel=None)
sns.boxplot(data=iqr_dist_cleaned, x='valence', color="red", ax=ax2)
plt.show()

# Conclusion: "automatic outlier procedures likely cause false positves and false negatives (moving the mouse around
# might actually be a sign of emotion). Different thresholds could be tried out (but also use the raw data or only
# exclude extreme long task duration datasets).
# Using a threshold of 3 * IQR flags about 1-2% of the data as outliers

