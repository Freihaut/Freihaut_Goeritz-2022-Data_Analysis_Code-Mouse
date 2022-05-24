'''
Code to transform the raw mouse usage data into a set of mouse-task features ready for data analysis
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''

# package import
import gzip
import json
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
from collections import defaultdict
import antropy
import seaborn as sns

#%%

# import the dataset file
# The task raw data is a dictionary that contains an entry for each study participant with the participants sociodems
# as well as the participants task data for each of his measurement participation
with gzip.open("Datasets_Raw/MouseTask_dataset.json.gz", "rb") as f:
    raw_data = json.loads(f.read())

#%%

# print some basic infos about the dataset to get a better "feeling" for it
print(f"Number of participants in the dataset: {len(raw_data)}")
print(f"Number of individual measurements in the study: {sum([len(raw_data[i]['TaskData']) for i in raw_data])}, Note"
      f" that this number includes measurements which were logged twice due to technical problems")


#%%

# Before any analysis, get descriptive stats about the number of logged measurements in the dataset
# Loop the raw dataset and create the analysis dataset with all the relevant variables
measurements_per_participant = {}

# loop all participants
for par in raw_data:
    measurements_per_participant[par] = {}
    # get the length of the logged task data
    num_datasets = len(raw_data[par]["TaskData"])

    # if it is the appVersion 1.0, some recordings need to be removed because they were logged twice
    if raw_data[par]["Sociodem"]["appVersion"] == "1.0":
        # to do so, loop all of the participants taskData, get the timestamp of the first task datapoint and delete
        # entries with duplicated timestamps
        task_info = {}
        for i in raw_data[par]["TaskData"]:
            task_info[i] = {}
            task_info[i]["first_t"] = raw_data[par]["TaskData"][i]["mTaskData"]["0"]["t"]

        task_inf_df = pd.DataFrame(task_info).T
        # print(task_inf_df)
        task_inf_df.drop_duplicates(inplace=True)
        num_datasets = len(task_inf_df)

    measurements_per_participant[par] = {"num_measures": num_datasets}

measurements_per_participant_df = pd.DataFrame(measurements_per_participant).T

# get basic descriptive stats about the number of measurements per participant
print(f"Total Number of measurements after removing duplicated ones: {measurements_per_participant_df['num_measures'].sum()}")
print(f"Descriptive Stats about the number of data measurements per participant:\n{measurements_per_participant_df.describe()}")

#%%

########################################################
#### Mouse Task Data Preprocessing Helper Functions ####
########################################################

# The following section contains the helper functions to process the raw mouse data into mouse usage features
# which are 'analysis ready'.

# The raw mouse data is a dictionary which contains dictionaries for each generated datapoint during the task
# This function processes the raw data dictionary into a format that is used in further processing steps,
# removes measurement artifacts and logs some recording quality criteria
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

    # get the number of artifacts in percent (if there is no movement data, set the value to nan
    try:
        artifact_percentage = (artifacts / total_movement_points) * 100
    except:
        artifact_percentage = np.nan

    # get the median time difference between consecutive mouse data points (if this number is too high, the sample
    # frequency is too low to record mouse movement adequately
    # get the median time difference between consecutive mouse datapoints to get the sampling frequency
    median_time_diff = np.median(np.diff([i["t"] for i in clean_list]))

    # return the cleaned mouse data list and the percentage of artifacts
    return clean_list, artifact_percentage, median_time_diff, valid_movement_datapoints


# interpolation function to resample the datapoints into equally long timestamps. Relevant in domains with different
# sampling frequencies (Wulff, D. U., Kieslich, P. J., Henninger, F., Haslbeck, J. M. B., & Schulte-Mecklenbeck, M.
# (2021, December 23). Movement tracking of cognitive processes: A tutorial using mousetrap. https://doi.org/10.31234/osf.io/v685r)
# which is the case in our study

# input data is a dataframe of mouse movement datapoints, an interpolation constant can be used to vary the time diff
# between consecutive datapoints
def interpolate_mouse_movement(data, interpolation_constant=15):

    # creates an interpolation function using the x- & y-coordinate and the timeline
    inter_x = scipy.interpolate.interp1d(data["t"], data["x"])
    inter_y = scipy.interpolate.interp1d(data["t"], data["y"])

    # create a new timeline array with equal timesteps using the start and endpoints
    equal_time_intervals = np.arange(data["t"].iloc[0], data["t"].iloc[-1], interpolation_constant)

    # return a dataframe with the interpolated datapoints
    inter_df = pd.DataFrame({"inter_x": np.round(inter_x(equal_time_intervals), decimals=3),
                             "inter_y":  np.round(inter_y(equal_time_intervals), decimals=3),
                             "inter_t": equal_time_intervals})

    return inter_df

#%%

############################################################
## helper functions to calculate the mouse usage features ##
############################################################


def get_movement_parameters(dframe):

    results = {}

    # calculate the euclidean distance between the datapoints
    euclidean_dist = np.sqrt(np.diff(dframe["inter_x"])**2 + np.diff(dframe["inter_y"])**2)

    # calculate the euclidean distance between the first and last datapoint (=optimal line between start and end point)
    shortest_distance = np.sqrt((dframe["inter_x"].iloc[-1] - dframe["inter_x"].iloc[0])**2 +
                                (dframe["inter_y"].iloc[-1] - dframe["inter_y"].iloc[0])**2)

    # simple plot of the shortest distance versus the actual distance
    # plt.plot(dframe["inter_x"], dframe["inter_y"], color="blue")
    # plt.plot(np.array([dframe["inter_x"].iloc[-1], dframe["inter_x"].iloc[0]]),
    #          np.array([dframe["inter_y"].iloc[-1], dframe["inter_y"].iloc[0]]), color="orange")
    # plt.show()

    # calculate the total travelled mouse distance as the sum of all euclidean distances
    total_distance = np.sum(euclidean_dist)
    # calculate the "overshoot" between the ideal line between start and end and the total distance
    distance_overshoot = total_distance / shortest_distance

    # calculate the movement speed in pixel per second
    delta_time = np.diff(dframe["inter_t"])
    speed = (euclidean_dist / delta_time) * 1000

    # calculate the movement acceleration in pixel per second²
    accel = (np.diff(speed) / delta_time[:-1])

    # calculate jerk, the rate-of-change of acceleration in pixel per second³
    # see https://en.wikipedia.org/wiki/Jerk_(physics)
    jerk = (np.diff(accel) / delta_time[:-2])

    results["total_dist"] = total_distance
    results["distance_overshoot"] = distance_overshoot
    results["speed_mean"] = np.mean(speed)
    results["speed_sd"] = np.std(speed)
    results["abs_accel_mean"] = np.mean(abs(accel))
    results["abs_accel_sd"] = np.std(abs(accel))
    results["abs_jerk_mean"] = np.mean(abs(jerk))
    results["abs_jerk_sd"] = np.std(abs(jerk))

    return results


def get_movement_angles(dframe):

    # get the vectors as the difference between consecutive datapoints
    movement_vectors = np.stack((np.diff(dframe["inter_x"]), np.diff(dframe["inter_y"])), axis=-1)

    # calculate the angles between consecutive vectors of mouse datapoints
    # using the numba decorator speeds the code up by a lot if the dataset gets bigger, but if the dataset it <= 15k
    # datapoints, the numba approach is slower. The numpy approach here is faster than the old loop approach
    # @numba.njit(fastmath=True)
    def _calc_angles(vector_matrix):

        angles = np.empty(vector_matrix.shape[0] - 1)

        for i in range(vector_matrix.shape[0] - 1):

            dot_prod = np.dot(vector_matrix[i], vector_matrix[i + 1])

            determinant = np.linalg.det(vector_matrix[i:i+2])

            angles[i] = np.arctan2(determinant, dot_prod)

        angles = np.absolute(angles)
        angles = np.degrees(angles)

        return angles

    movement_angles = _calc_angles(movement_vectors)

    return {"angle_mean": np.mean(movement_angles), "angle_sd": np.std(movement_angles)}


def get_x_y_flips(dframe):

    def _get_flips(array):
        pos = array > 0
        npos = ~pos
        return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0].shape[0]

    x_shifts = np.diff(dframe["inter_x"])
    y_shifts = np.diff(dframe["inter_y"])

    x_flips = _get_flips(x_shifts[x_shifts != 0])
    y_flips = _get_flips(y_shifts[y_shifts != 0])

    return {"x_flips": x_flips, "y_flips": y_flips}


# Entropy is not calculated any more, because it is a hard to interpret parameter, has two tuning options and produces
# NaN values for the trial data

# function to calculate the entropy of a trajectory as a measure of the complexity of movement along one specific dimension
# The parameter was propsed by:
# Hehman, E., Stolier, R. M., & Freeman, J. B. (2015). Advanced mouse-tracking analytic techniques for enhancing
# psychological science. Group Processes & Intergroup Relations, 18(3), 384-401.
# The function was written to mimic the entropy function of the mousetrap package (the results are the same):
# https://github.com/PascalKieslich/mousetrap/blob/master/R/sample_entropy.R
# A (way) faster alternative solution is the sample_entropy function of the antropy package, which is used here:
# https://raphaelvallat.com/antropy/build/html/index.html
# e.g. antropy.sample_entropy(data, order=3, metric='chebyshev') calculates the same entropy result
# this function calculates entropy across both, the x-movement axis and the y-movement axis
def get_x_y_entropy(dframe):

    def get_entropy(L, m, r):

        Mm = 0.0
        Mm1 = 0.0

        # check if the number of datapoints is sufficient
        if len(L) >= (m + 2):
            def _windowmaker(data, window_size):
                wind = np.array([data[i:(i+window_size)] for i in range(len(data) - window_size + 1)])

                return wind

            wm = _windowmaker(L, m)
            wm = wm[:-1]
            wm1 = _windowmaker(L, m+1)

            for i in range(wm.shape[0] - 1):
                for j in range(i+1, wm.shape[0]):
                    if max(abs(wm[i, :] - wm[j, :])) <= r:
                        Mm += 1
                        if max(abs(wm1[i, :] - wm1[j, :])) <= r:
                            Mm1 += 1

            # check if is a zero division error (Mm = 0)
            try:
                samp_ent = -np.log((Mm1 / Mm))
            except ZeroDivisionError:
                samp_ent = np.nan
        else:
            samp_ent = np.nan

        return samp_ent

    # calculate the entropy for the movement differences across the x-axis
    x_diffs = np.diff(dframe["inter_x"])
    # calc the entropy according to the parameters recommended by Hehman
    x_entroy = get_entropy(x_diffs, 3, .2 * np.std(x_diffs, ddof=1))
    print(x_entroy)
    # use the antropy calculation: it needs an error check if there is not enough data
    try:
        x_entroy = antropy.sample_entropy(x_diffs, 3, metric='chebyshev')
    except IndexError:
        x_entroy = np.nan
    # repeat the same for the y-axis movement
    y_diffs = np.diff(dframe["inter_y"])
    # y_entroy = get_entropy(y_diffs, 3, .2 * np.std(y_diffs, ddof=1))
    try:
        y_entroy = antropy.sample_entropy(y_diffs, 3, metric='chebyshev')
    except IndexError:
        y_entroy = np.nan

    print(x_entroy)
    # return the entropy values
    return {"x_entropy": x_entroy, "y_entropy": y_entroy}

#%%

##################################
#### Dataset Creation Process ####
##################################


# helper function to calculate all mouse usage features for a given dataset
# the expected input is a pandas dataframe of the mouse task data
def calculate_mouse_parameters_pipeline(mouse_data):

    # empty dict to store the calculated parameters
    mouse_features = {}

    # check if there are enough movement datapoints in the dataset, there should be at least 5 per trial (which is
    # a very liberal criteria
    if len(mouse_data.loc[mouse_data["e"] == "Mo"]) <= 30:
        return mouse_features

    # Task based mouse features
    # Time it
    # start = timer()

    # calculate the task duration as the time difference between the first and last datapoint of the task in seconds
    task_duration = (mouse_data["t"].iloc[-1] - mouse_data["t"].iloc[0]) / 1000
    mouse_features["task_duration"] = task_duration

    # get the number of mouseclicks
    mouse_clicks = len(mouse_data.loc[mouse_data["e"] == "Cl"])
    mouse_features["clicks"] = mouse_clicks

    # isolate the movement data (all features are calculated based on the mouse movement), clicks are excluded, except
    # for the first (click) datapoint, which is the start position of the mouse
    movement_data = mouse_data.loc[mouse_data["e"] == "Mo"]
    # add first trial datapoint
    movement_data = pd.concat([mouse_data.head(1), movement_data], ignore_index=True)

    # interpolate the movement data to a constant time diff interval of 15ms between consecutive movement datapoints
    interpol_data = interpolate_mouse_movement(movement_data, interpolation_constant=15)
    # there are cases where the first click datapoint of the trial and the first move datapoint have the same
    # timestamp, which causes the interpolation function to throw a warning and a NaN value for the first
    # datapoint, remove nans from the interpol data
    interpol_data.dropna(inplace=True)

    # simple visualization of the original data vs the interpolated data
    # plt.plot(mouse_data["x"], mouse_data["y"], color="blue")
    # plt.plot(interpol_data["inter_x"], interpol_data["inter_y"], color="orange")
    # plt.show()

    # calculate the task based mouse usage features and add them to the mouse features dictionary
    task_based_features = {}
    # start = timer()
    task_based_features.update(get_movement_parameters(interpol_data))
    # delete the task overshoot feature because it does not make sense for the entire task
    task_based_features.pop('distance_overshoot', None)
    # print(f"Move Params elapsed: {timer() - start}")
    # start = timer()
    task_based_features.update(get_movement_angles(interpol_data))
    # print(f"Angles elapsed: {timer() - start}")
    start = timer()
    task_based_features.update(get_x_y_flips(interpol_data))
    # print(f"Flips: {timer() - start}")
    # start = timer()
    # task_based_features.update(get_x_y_entropy(interpol_data)) # dont use entropy
    # print(f"Entropy: {timer() - start}")
    for feat in task_based_features:
        mouse_features["task_" + feat] = task_based_features[feat]

    # print(f"Time to calc task params: {timer() - start}")

    # time the trial calculations
    # start = timer()

    # Trial based mouse features
    # to calculate trial based features, separate the dataset into the trials and calculate the features per trial
    # create a default dict to store the features in a list
    trial_dict = defaultdict(list)
    # loop over the trials
    for i in range(1, 7):
        # isolate the trial dataset
        trial_data = mouse_data.loc[mouse_data["cN"] == i]

        # check if there is enough data in the trial movement dataset, the 5 datapoints threshold is set very low
        # a proper recording of a trial likely has at least 10 datapoints, most have more than 20
        if len(trial_data.loc[trial_data["e"] == "Mo"]) > 5:

            # get the trial duration (time between the last and first datapoint of a trial)
            # the first datapoint of a trial is the click on the previous point and the last datapoint is the last
            # movement before the next point is clicked
            trial_duration = (trial_data["t"].iloc[-1] - trial_data["t"].iloc[0]) / 1000
            trial_dict["duration"].append(trial_duration)

            # get the trial movement data, but include the first (click) datapoint as this represents the starting
            # position of each trial
            # extract all move datapoints
            trial_move_data = trial_data.loc[trial_data["e"] == "Mo"]
            # add first trial datapoint
            trial_move_data = pd.concat([trial_data.head(1), trial_move_data], ignore_index=True)

            # calculate the trial offset time, the time difference between the last movement and first movement of
            # consecutive trials (time difference between moving to the point to click and start moving to the next
            # point after the click): this parameter cant be calculated for the last trial
            if i < 6:
                last_trial_timestamp = trial_move_data["t"].iloc[-1]
                next_trial_data = mouse_data.loc[(mouse_data["cN"] == i + 1)]
                first_next_trial_timestamp = next_trial_data.loc[next_trial_data["e"] == "Mo"]["t"].iloc[0]
                trial_dict["trial_move_offset"].append(first_next_trial_timestamp - last_trial_timestamp)

            # interpolate the movement data and calculate the movement features
            trial_interpol = interpolate_mouse_movement(trial_move_data, interpolation_constant=15)
            # there are cases where the first click datapoint of the trial and the first move datapoint have the same
            # timestamp, which causes the interpolation function to throw a warning and a NaN value for the first
            # datapoint, remove nans from the interpol data
            trial_interpol.dropna(inplace=True)

            # print(trial_interpol)
            # visualize the interpol vs the real data
            # sns.lineplot(x=trial_data["x"], y=trial_data["y"], color="orange")
            # sns.lineplot(x=trial_interpol["inter_x"], y=trial_interpol["inter_y"], color="blue")
            # plt.show()

            trial_features = {}
            trial_features.update(get_movement_parameters(trial_interpol))
            trial_features.update(get_movement_angles(trial_interpol))
            trial_features.update(get_x_y_flips(trial_interpol))
            # trial_features.update(get_x_y_entropy(trial_interpol)) # dont use entropy
            for feat in trial_features:
                trial_dict[feat].append(trial_features[feat])

    # calculate the mean and standard deviation of the trial features and add the to the mouse features dict
    # first remove the first and last datapoint
    for trial_feats in trial_dict:
        mouse_features["trial_mean_" + trial_feats] = np.nanmean(trial_dict[trial_feats])
        mouse_features["trial_sd_" + trial_feats] = np.nanstd(trial_dict[trial_feats])

    # print(f"Time to calc trial params: {timer() - start}")

    # return all calculated mouse features
    return mouse_features

#%%


# Loop the raw dataset and create the analysis dataset with all the relevant variables
mouse_task_dataset = {}

# loop all participants
for par in raw_data:

    print(f"Processing Participant: {par}")

    # create a dictionary entry for the participant
    mouse_task_dataset[par] = {}

    # loop over all datasets of the participant
    for num, dset in enumerate(raw_data[par]["TaskData"]):

        # track the time it takes to process a dataset
        # start = timer()

        # create a dict entry for the measurement timepoint
        mouse_task_dataset[par][num] = {}

        # get task meta-information
        mouse_task_dataset[par][num]["timestamp"] = raw_data[par]["TaskData"][dset]["time"]
        mouse_task_dataset[par][num]["taskNum"] = raw_data[par]["TaskData"][dset]["taskInf"]["taskNum"]
        mouse_task_dataset[par][num]["zoom"] = raw_data[par]["TaskData"][dset]["disInf"]["zoom"]
        mouse_task_dataset[par][num]["screen_width"] = raw_data[par]["TaskData"][dset]["disInf"]["screenSize"]["width"]
        mouse_task_dataset[par][num]["screen_height"] = raw_data[par]["TaskData"][dset]["disInf"]["screenSize"]["height"]

        # get the mouse data
        mouse_task_data = raw_data[par]["TaskData"][dset]["mTaskData"]

        # clean the dataset and get data quality infos
        clean_data, artifacts, median_time_diff, move_points = clean_mouse_task_data(mouse_task_data)

        # save the data quality info of the measurement timepoint
        mouse_task_dataset[par][num]["artifact_percent"] = artifacts
        mouse_task_dataset[par][num]["median_sampling_freq"] = median_time_diff
        mouse_task_dataset[par][num]["move_datapoints"] = move_points

        # calculate all mouse features (the cleaned mouse data must be transformed into a pandas dataframe)
        all_mouse_features = calculate_mouse_parameters_pipeline(pd.DataFrame(clean_data))
        mouse_task_dataset[par][num].update(all_mouse_features)

        # get the arousal and valence values of the task
        valence = raw_data[par]["TaskData"][dset]["selfReportData"]["valence"]
        arousal = raw_data[par]["TaskData"][dset]["selfReportData"]["arousal"]
        mouse_task_dataset[par][num]["valence"] = valence
        mouse_task_dataset[par][num]["arousal"] = arousal
        # calculate the stress variable (if valence and arousal values are < 50)
        mouse_task_dataset[par][num]["stress"] = 1 if arousal > 50 and valence < 50 else 0

        # get the sociodemographics of the participant
        mouse_task_dataset[par][num]["age"] = raw_data[par]["Sociodem"]["age"]
        mouse_task_dataset[par][num]["sex"] = raw_data[par]["Sociodem"]["sex"]
        mouse_task_dataset[par][num]["hand"] = raw_data[par]["Sociodem"]["hand"]
        # save the participant ID
        mouse_task_dataset[par][num]["ID"] = par
        # add the dataset ID (for data quality analysis in order to easily extract the raw dataset of the measurment)
        mouse_task_dataset[par][num]["dset_ID"] = dset

        # print(f"Elapsed time to process the dataset: {timer() - start}")


# convert the dictionary into a dataframe that has a row for each measurement timepoint
mouse_task_dataset_df = pd.concat({k: pd.DataFrame(v).T for k, v in mouse_task_dataset.items()}, axis=0).round(4).reset_index(drop=True)

#%%

print(f"Shape of the untreated dataframe: {mouse_task_dataset_df.shape}")

# mouse_task_dataset_df.reset_index(drop=True, inplace=True)

#%%

###############################
#### Removal of bad trials ####
###############################

# before the dataset is ready for analysis, some trials need to be removed. This removal procedure tries to eliminate
# only trials with technical problems, i.e. the mouse data logging did not work as intended
# (trials with potential unwanted task behavior, e.g. participants took a break or randomly moved the mouse around
# during the task are elimnated in a later step, see the data analysis code)

# Error 1: There was a bug that caused some trials to get saved multiple times: Remove all but one of those trials

# all variables except for the logged time should be identical
all_columns = list(mouse_task_dataset_df.columns)
# remove the data saving timestamp and the dataset ID
all_columns.remove("timestamp")
all_columns.remove("dset_ID")

mouse_task_dataset_df.drop_duplicates(subset=all_columns, inplace=True)
# This is the "original number of logged trials across all participants"
print(f"Shape of dataset after removal of duplicated trials: {mouse_task_dataset_df.shape}")

# save the dataframe for data quality analysis -> This is the "raw features data"
mouse_task_dataset_df.to_csv("Mouse_Task_Features_Data_Quality_Inspection.csv", index=False)

#%%

mouse_task_dataset_df = pd.read_csv("Mouse_Task_Features_Data_Quality_Inspection.csv")

print(len(mouse_task_dataset_df["ID"].unique()))

#%%

# Error 2: Identify and remove cases with technical logging problems

# exclude cases in which the logging interval was not good enough to accurately sample mouse usage during the task
# the cut-off median logging interval was set to ~50ms (20 Hz). Additionally, remove cases where too few mouse datapoints
# were recorded (< 88 datapoints). The cut-offs were chosen based on visual inspection of "problematic cases" (see
# the Mouse_Task_Data_Quality_Analysis Script).
# For most of these cases, no mouse usage parameters were calculated, because too few datapoints caused errors in the
# parameter calculation code
# (see the data processing code)
problematic_logging = mouse_task_dataset_df.loc[(mouse_task_dataset_df["median_sampling_freq"] > 46) |
                                                (mouse_task_dataset_df["move_datapoints"] < 88)]
# all of these trials have 10 or fewer datapoints, the trial with the fewest datapoints out of the included trials
# has 88 datapoints

# drop the cases from the dataset
mouse_task_dataset_df = mouse_task_dataset_df.drop(problematic_logging.index)

print(f"Number of bad mouse logging trials: {len(problematic_logging)}")
print(f"Shape of dataset after removal of trials with a bad mouse logging: {mouse_task_dataset_df.shape}")

#%%

# Remove participants with too little data (= participated in too little data collections).
# A few participants only took part in 3 or less data collections. Especially machine learning analysis require that
# there are at least 6 measurement timepoints per participant to be able to split them into a training and test subsample
# we therefore decided to remove those participants with 3 or less completed data collections
# before their deletion, check their stress level and compare it to the entire sample, if there could be a bias that
# stresses participants could not participate often

# add the number of timepoints per participant to the dataframe
mouse_task_dataset_df["freq"] = mouse_task_dataset_df.groupby("ID")["ID"].transform("count")
# get participants with low number of participations
low_participation = mouse_task_dataset_df.loc[mouse_task_dataset_df["freq"] <= 3]
print(f"Number of participants with 3 or less datasets: {low_participation['ID'].nunique()}")
print(f"Number of datasets from participants with 3 or less datasets: {len(low_participation)}")
mouse_task_dataset_df = mouse_task_dataset_df.loc[mouse_task_dataset_df["freq"] > 3]
# compare their stress level with the stress level of the sample with more measurement timepoints
print(f"Number of Stressful measurements compared to the total sample for the low participation dataset "
      f"{len(low_participation.loc[low_participation['stress'] == 1]) / len(low_participation) * 100}%")
print(f"Number of Stressful measurements compared to the total sample for the cleaned dataset "
      f"{len(mouse_task_dataset_df.loc[mouse_task_dataset_df['stress'] == 1]) / len(mouse_task_dataset_df) * 100}%")


print(f"Shape of dataset after removal of participants with too little trials: {mouse_task_dataset_df.shape}")


#%%

# save the dataset as a csv file for data analysis
mouse_task_dataset_df.to_csv("Mouse_Task_Features.csv", index=False)


#%%

################
# Extract raw data and the calculated mouse features for a sample participant to validate the feature calculation
# functions against the feature calculation of the mousetrap package
################

# get a sample participant dataset
sample_participant_data = raw_data['2paEfrfU0p']["TaskData"]['-MkhYs-ufNETCZIXAAQz']["mTaskData"]

# clean the data
clean_par_dat, par_art, par_log_int, par_val_movement = clean_mouse_task_data(sample_participant_data)
# convert cleaned data to a dataframe
cleaned_dat_df = pd.DataFrame(clean_par_dat)

# calculate the mouse usage features
par_features = calculate_mouse_parameters_pipeline(cleaned_dat_df)

# get the raw mouse movement data (with the click as the starting point of the trial),
# which is used to calculate the movement related mouse features
trial_movements = []
# loop the data by trial
for i in range(1, 7):
    # get the trial movement data plus the first datapoint
    trial_dat = pd.concat([cleaned_dat_df.loc[cleaned_dat_df["cN"] == i].head(1),
                           cleaned_dat_df.loc[(cleaned_dat_df["cN"] == i) & (cleaned_dat_df["e"] == "Mo")]],
                          ignore_index=True)
    # add it to the trial movement list
    trial_movements.append(trial_dat)

# concat all trials together
par_movement_df = pd.concat(trial_movements, ignore_index=True)

# save the calculated features and dataframe as a json file for import in R
mousetrap_validation_data = {"Py_Mouse_Features": par_features, "Raw_Data": par_movement_df.to_dict()}
with open("Mousetrap_Validation_Data.json", "w") as fp:
    json.dump(mousetrap_validation_data, fp)


#%%

#################
#### Testing ####
#################

# get test data of a participant to try out the preprocessing functions

test_data = raw_data['6oQ5w5Dui8']["TaskData"]['-Mnl-aAKBJwKUZ_1YhRL']["mTaskData"]

clean_list, artifact_percentage, median_time_diff, valid_movement_datapoints = clean_mouse_task_data(test_data)

test2 = pd.DataFrame(clean_list)

a = calculate_mouse_parameters_pipeline(test2)


#%%

# inspect case with interpolation warning
for i in raw_data['5ck62p5ZfW']["TaskData"]:

    if (i == "-MkqDy8BOIsE_YLzEIVC"):
        print(i)
        m_dat = raw_data['5ck62p5ZfW']["TaskData"][i]["mTaskData"]
        clean_list, artifact_percentage, median_time_diff, valid_movement_datapoints = clean_mouse_task_data(m_dat)
        calculate_mouse_parameters_pipeline(pd.DataFrame(clean_list))

#%%

# create some pictures of the mouse usage during different tasks

for k, i in enumerate(raw_data['2paEfrfU0p']["TaskData"]):

    print(i)

    mdata = raw_data['2paEfrfU0p']["TaskData"][i]["mTaskData"]

    dat, b, c, d = clean_mouse_task_data(mdata)

    dat = pd.DataFrame(dat)

    # visualize the task data and save the image
    colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'}

    plt.scatter(dat["x"], dat["y"], color=dat["cN"].map(colors), marker=".")
    grouped = dat.groupby("cN")
    for key, group in grouped:
        plt.plot(group["x"], group["y"], color=colors[key])

    plt.title("Task Number: " + str(raw_data['2paEfrfU0p']["TaskData"][i]["taskInf"]["taskNum"]))
    # plt.savefig("TaskImg_" + str(raw_data['2paEfrfU0p']["TaskData"][i]["taskInf"]["taskNum"]) + ".png")
    plt.show()

    if k >= 2: break
