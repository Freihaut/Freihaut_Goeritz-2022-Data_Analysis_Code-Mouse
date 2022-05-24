'''
Code to transform the raw mouse usage data into a set of free-mouse features ready for data analysis
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''

# package import
import json
import gzip
import pandas as pd
import numpy as np
import scipy.interpolate
from collections import defaultdict
from timeit import default_timer as timer

#%%

# dataset import
# The free mouse usage data is a dictionary that contains an entry of the recorded free mouse usage data for each study
# participant with the participants sociodems
# The file is rather large because each trial (about 10.000 in total) contains up to 15.000 recorded datapoints
# loading the dataset takes some time
with gzip.open("Datasets_Raw/FreeMouse_dataset.json.gz", "rb") as f:
    raw_data = json.loads(f.read())


#%%

# Before any analysis, get descriptive stats about the number of logged measurements in the dataset
# Loop the raw dataset and create the analysis dataset with all the relevant variables
measurements_per_participant = {}
# a bug in an early version of the Study-App caused that data was saved multiple times in the database. We need to
# filter out the duplicates
duplicate_ids = []

# loop all participants
for par in raw_data:
    measurements_per_participant[par] = {}
    # get the length of the logged task data
    num_datasets = len(raw_data[par]["freeMouse"])

    # if it is the appVersion 1.0, some recordings need to be removed because they were logged twice
    if raw_data[par]["Sociodem"]["appVersion"] == "1.0":
        # to do so, loop all of the participants taskData, get the timestamp of the first task datapoint and delete
        # entries with duplicated timestamps
        task_info = {}
        for i in raw_data[par]["freeMouse"]:
            task_info[i] = {}
            task_info[i]["first_t"] = raw_data[par]["freeMouse"][i]["mFreeUseData"]["0"]["t"]

        task_inf_df = pd.DataFrame(task_info).T
        # get the duplicated rows and save them in order to exclude them for later calculations
        duplicates = task_inf_df.duplicated()
        duplicates = list(duplicates.loc[duplicates].index)
        duplicate_ids.extend(duplicates)
        # print(task_inf_df)
        # drop the duplicates
        task_inf_df.drop_duplicates(inplace=True)
        num_datasets = len(task_inf_df)

    measurements_per_participant[par] = {"num_measures": num_datasets}

measurements_per_participant_df = pd.DataFrame(measurements_per_participant).T

# get basic descriptive stats about the number of measurements per participant

# print basic information about the dataset
print(f"Number of participants in the dataset: {len(measurements_per_participant_df)}")
print(f"Total Number of measurements after removing duplicated ones: "
      f"{measurements_per_participant_df['num_measures'].sum()}")
print(f"Number of removed duplicated trials: {len(duplicate_ids)}")
print(f"Descriptive Stats about the number of data measurements per participant:\n"
      f"{measurements_per_participant_df.describe()}")


#%%

##################################################
# Mouse Task Data Preprocessing Helper Functions #
##################################################

# The following section contains the helper functions to process the raw mouse data into mouse usage features
# which are 'analysis ready'. The helper functions are similar to the helper functions for processing the
# mouse task data. However, there are some key differences between the free mouse usage data and the task data
# which requires different preprocessing steps

# clean the free mouse usage data
def clean_free_mouse_data(mouse_data):

    # first, sort the dictionary by the timestamp to get the recorded mouse datapoints chronologically
    # the output is a list of tuples. The first tuple entry is the number, the second is the datapoint
    sorted_mouse_data = sorted(mouse_data.items(), key=lambda x: x[1]["t"])

    # Free mouse usage data logging started 5 minutes prior to the pop-up of a data collection window and ended
    # as soon as participants clicked on the first target in the point-and-click task. This caused the logging interval
    # of the recorded mouse usage to be of unequal length (because participants did not immediately work on the
    # point and click task). The Study-App had a limit of logged 15.000 logged datapoints (if the limit was exceeded
    # adding a new datapoint removed the oldest datapoint). The limit was set because the App recorded mouse usage data
    # at a sampling rate of 50Hz (a datapoint was logged every 20ms), which results in 15.000 datapoints per 5 minutes.
    # However, sampling frequencies were not consistent, which requires selecting the 5 minute interval from the data
    # To do so, find the position of the first timepoint that has a shorter than 5 minute time difference to the last
    # datapoint:
    last_recorded_timestamp = sorted_mouse_data[-1][1]["t"]
    for critical_index, datapoint in enumerate(sorted_mouse_data):
        # stop the loop if the time difference between the last datapoint and current datapoint is smaller than 5
        # minutes (= 300000ms = 5 * 60 * 1000)
        if datapoint[1]["t"] - last_recorded_timestamp > -300000:
            break

    # select the datapoints after greater or equal than the critical index
    five_minute_interval = sorted_mouse_data[critical_index:]

    # convert the data into a dataframe
    mouse_df = pd.DataFrame([i[1] for i in five_minute_interval])

    # delete artifacts = datapoints with timestamps that already exist, so drop rows with duplicated timestamps
    mouse_df = mouse_df.drop_duplicates(subset=['t'])

    # some of the recorded datapoints have very large negative x-& y-values. These datapoints must be converted
    # to the be the same value as the last valid datapoint (removing them would cause some
    # trials to drop entirely. The large negative values seem to happen if the computer goes into the lock screen,
    # which could be seen as a pause, where no movement happens. Movement on the lockscreen will also be labeled
    # as a pause, because lockscreen mouse values always have the same x-& y- coordinate)
    # Note that there are also valid negative screen values, which are recorded if the user has multiple monitors
    # see: https://github.com/electron/electron/issues/22659
    # they should not be removed

    # get the lockscreen time (lockscreen values are set to -10,000 as the cutoff, observed values are much lower,
    # but -10,000 should not conflict with valid lockscreen values)
    lockscreen_df = mouse_df.copy()
    lockscreen_df["lock"] = np.where((lockscreen_df["x"] < -10000) | (lockscreen_df["y"] < -10000), 1, 0)

    # number all lockscreen episodes
    lock_time = 0
    lock_ep_num = 0
    lock_started = False
    for k, i in enumerate(lockscreen_df['lock']):
        # if it is a pause
        if i == 1:
            # if the previous value wasnt a pause already
            if not lock_started:
                # increase the pause number, get the start time of the pause
                lock_ep_num += 1
                lock_started = lockscreen_df["t"].iloc[k]
        else:
            # if it is a movement datapoint and there was a lockscreen time before, stop the lockscreen pause
            # and add the lockscreen time
            if lock_started:
                lock_time += lockscreen_df["t"].iloc[k - 1] - lock_started
                lock_started = False

    # replace the lockscreen values with a previous valid value or 0 if there is no previous valid value (if mouse
    # usage recording starts with lockscreen values)
    mouse_df = mouse_df.mask(mouse_df.lt(-10000)).ffill().fillna(0)

    # get artifact percentage (datapoints with doubled timestamp values)
    artifact_percentage = (1 - len(mouse_df) / len(five_minute_interval)) * 100

    return mouse_df, artifact_percentage, {'lockscreen_episodes:': lock_ep_num, 'lockscreen_time': lock_time}


# isolate movement episodes and pauses (based on a threshold value that defines a pause)
# the function returns multiple dataframes if multiple pause threshold values are passed in it
# this is done in order to optimize the runtime (the runtime is shorter if the function returns all threshold datasets
# at once instead of running the entire function multiple times for different pause threshold values)
# TODO: This function is running multiple for loops and should be optimized
def get_movement_episodes(mouse_data, pause_thresholds):

    # get the difference values between the mouse usage coordinates and fill the first na datapoint
    diff_data = mouse_data.diff().bfill()

    # select movement episodes = episodes with changes in x- & y-values
    diff_data["mo"] = np.where((diff_data["x"] != 0) | (diff_data["y"] != 0), 1, 0)

    # number pauses (all episodes with no movement change)
    numbered_pauses = []
    pause_num = 0
    pause_started = False
    for k, i in enumerate(diff_data['mo']):
        # if it is a pause
        if i == 0:
            # if the previous value wasnt a pause already
            if not pause_started:
                # increase the pause number, start the pause
                pause_num += 1
                pause_started = True
            # add the pause number to the list with all pause values
            numbered_pauses.append(pause_num)
        else:
            # if it is a movement datapoint, stop the pause and add 0 to the numbered pause list (0 = movement)
            if pause_started:
                pause_started = False
            numbered_pauses.append(0)

    # add the numbered pause list to the difference data
    diff_data["num_pauses"] = numbered_pauses

    # get the length of all marked pauses
    pause_length = diff_data.groupby(["num_pauses"]).sum()['t'][1:]

    # select "valid" pauses (no movement for a time that is longer than a specified threshold)
    # get the valid pauses for every specified pause threshold
    valid_pauses = {}
    for pause_threshold in pause_thresholds:
        valid_pauses[str(pause_threshold)] = pause_length.index[pause_length > pause_threshold].values

    # count number of movements between valid pauses
    # if smaller than 3, it was no movement, else, it was movement
    # label all valid movement episodes between valid pauses (valid movement episodes require at least 3 datapoints
    # with changes in the mouse cursor position)
    # do this for every pause threshold and create different movement episode datasets:
    mo_episode_dsets = {}
    # loop all pauses
    for valid_pause in valid_pauses:
        # intitate a bool that keeps track of the pauses
        pause_started = False
        # initiate a counter for movement datapoints
        movements = 0
        # initiate a list that holds all numbered movement episodes
        dpoints = []
        # initiate a movement episode counter
        move_classifier = 1
        # initiate a list that holds movement episode data
        episode = []
        # loop all numbered pause values
        for i in diff_data["num_pauses"]:
            # check if the numbered pause is a "real" pause defined on the pause threshold value
            # if it is not a real pause
            if i not in valid_pauses[valid_pause]:
                # switch the pause tracker bool
                pause_started = False
                # check if it is a movement datapoint, if so increment the movement datapoint counter
                if i == 0:
                    movements += 1
                    episode.append(move_classifier)
                else:
                    episode.append(move_classifier)
            # if it is a "real pause"
            else:
                # if the pause tracker bool hasnt been switched (if it is the first valid pause datapoint)
                if not pause_started:
                    # switch it
                    pause_started = True
                    # check if the previous movement data episode contains more than 3 movement datapoints, if it does
                    # add the movement episodes data as valid movement data to the dpoints list, reset the episodes
                    # list and the movements datapoint counter and incremeant the movement episode label
                    if movements > 3:
                        dpoints.extend(episode)
                        episode = []
                        movements = 0
                        move_classifier += 1
                    # if the movement episode does not contain enough movement data ( = less than 3 movement datapoints)
                    # add the movement episode as a "pause", reset the episode list and the movement episodes counter
                    else:
                        dpoints.extend(len(episode) * [0])
                        episode = []
                        movements = 0
                # add a pause datapoint to the dpoints list
                dpoints.append(0)
        # after the loop, add the last movement episodes list to the dpoint list if it includes more than 3 movement
        # datapoints, otherwise include the last movement episode as a pause episode
        if movements > 3:
            dpoints.extend(episode)
        else:
            dpoints.extend(len(episode) * [0])
        # create a dataframe that includes the movement episodes indicator based on the specified pause threshold
        mouse_data_episode = mouse_data.copy()
        # add the dpoints to the copy of the mouse data
        mouse_data_episode["mo_episodes"] = dpoints
        mo_episode_dsets[valid_pause] = mouse_data_episode

    return mo_episode_dsets


# interpolation function to resample the datapoints into equally long timestamps. Relevant in domains with different
# sampling frequencies (Wulff, D. U., Kieslich, P. J., Henninger, F., Haslbeck, J. M. B., & Schulte-Mecklenbeck, M.
# (2021, December 23). Movement tracking of cognitive processes: A tutorial using mousetrap.
# https://doi.org/10.31234/osf.io/v685r)
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


# short helper function to get information about the logging interval (the time difference between consecutive mouse
# datapoints (in theory, the app logged a datapoint every 20ms, but in practice, the value could be different)
# the expected input is the cleaned mouse dataframe
def logging_interval_info(mouse_data):

    # get the time differences between consecutive mouse datapoints
    logging_intervals = np.diff(mouse_data["t"])

    # return some infos about the logging intervals (median interval, the maximal interval, and the standard deviation
    # of logging intervals
    return {
        "median_log_int": np.median(logging_intervals),
        "max_log_int": np.max(logging_intervals),
        "sd_log_int": np.std(logging_intervals)
    }


#%%

##########################################################
# Helper functions to calculate the mouse usage features #
##########################################################

def get_movement_parameters(dframe):

    results = {}

    # calculate the euclidean distance between the datapoints
    euclidean_dist = np.sqrt(np.diff(dframe["inter_x"])**2 + np.diff(dframe["inter_y"])**2)

    # calculate the total travelled mouse distance as the sum of all euclidean distances
    total_distance = np.sum(euclidean_dist)

    # calculate the movement speed in pixel per second
    delta_time = np.diff(dframe["inter_t"])
    speed = (euclidean_dist / delta_time) * 1000

    # calculate the movement acceleration in pixel per second²
    accel = (np.diff(speed) / delta_time[:-1])

    # calculate jerk, the rate-of-change of acceleration in pixel per second³
    # see https://en.wikipedia.org/wiki/Jerk_(physics)
    jerk = (np.diff(accel) / delta_time[:-2])

    results["total_dist"] = total_distance
    results["speed_mean"] = np.mean(speed)
    results["speed_sd"] = np.std(speed)
    results["abs_accel_mean"] = np.mean(abs(accel))
    results["abs_accel_sd"] = np.std(abs(accel))
    results["abs_jerk_mean"] = np.mean(abs(jerk))
    results["abs_jerk_sd"] = np.std(abs(jerk))

    return results


# calculate the angle between consecutive movements
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


# calculate the changes in movement on the x-movement axis and the y-movement axis
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


#%%

############################
# Dataset Creation Process #
############################

# Feature Calculation Helper Pipeline
# -----------------------------------

# helper function to calculate all mouse usage features for the free mouse usage data
# the input data is a pandas datframe with the recorded mouse usage and the movement episode label
def calculate_mouse_parameters_pipeline(mouse_data):

    # empty dict to store the calculated parameters
    mouse_features = {}

    # calculate the duration of the recorded free mouse usage data (should be 5 minutes)
    recording_duration = (mouse_data["t"].iloc[-1] - mouse_data["t"].iloc[0]) / 1000
    mouse_features["recording_duration"] = recording_duration

    # get the number of movement episodes
    move_episodes = mouse_data["mo_episodes"].max()
    mouse_features["movement_episodes"] = move_episodes

    # Calculate the mouse usage features per mouse movement episode
    # instead of taking the entire recorded mouse usage dataset, we take the movement episodes from the dataset and
    # calculate the mouse usage (movement) features per movement episode
    # setup a timer
    # start = timer()
    # setup a dictionary for storing the mouse parameters per movement episode
    mo_episodes_dict = defaultdict(list)
    # loop over all movement episodes
    for i in range(1, move_episodes+1):

        # get the movement episode data
        mo_episode_data = mouse_data.loc[mouse_data["mo_episodes"] == i]

        # get the duration of the movement episode
        mo_episode_duration = (mo_episode_data["t"].iloc[-1] - mo_episode_data["t"].iloc[0]) / 1000
        mo_episodes_dict["episode_duration"].append(mo_episode_duration)

        # get the logging interval of the movement episode to potentially spot bad logging trials
        log_int_info = logging_interval_info(mo_episode_data)
        for i in log_int_info:
            mo_episodes_dict[i].append(log_int_info[i])

        # interpolate the movement data and calculate the movement features
        trial_interpol = interpolate_mouse_movement(mo_episode_data, interpolation_constant=15)

        # calculate the mouse usage features per movement episode
        episode_features = {}
        episode_features.update(get_movement_parameters(trial_interpol))
        episode_features.update(get_movement_angles(trial_interpol))
        episode_features.update(get_x_y_flips(trial_interpol))
        # add all features to the movement episodes dicti
        for feat in episode_features:
            mo_episodes_dict[feat].append(episode_features[feat])

    # print(f"Time for feature calculation: {timer() - start}")
    # calculate the mean and standard deviation of the trial features and add the to the mouse features dict
    for trial_feats in mo_episodes_dict:
        mouse_features["mo_ep_mean_" + trial_feats] = np.nanmean(mo_episodes_dict[trial_feats])
        mouse_features["mo_ep_sd_" + trial_feats] = np.nanstd(mo_episodes_dict[trial_feats])

    # calculate the total movement duration, the total movement distance and the no_movement_percentage
    mouse_features["movement_duration"] = sum(mo_episodes_dict['episode_duration'])
    mouse_features["movement_distance"] = sum(mo_episodes_dict['total_dist'])
    mouse_features["no_movement"] = (1 - (mouse_features["movement_duration"] / recording_duration)) * 100

    # return all calculated mouse features
    return mouse_features


#%%

# Dataset Creation Loop
# -----------------------------------

#TODO: The loop takes pretty long to run, a todo option would be to parallelize it

# Loop the raw dataset and create the analysis dataset with all the relevant variables
free_usage_dataset = {}

# loop all participants
for par in raw_data:

    print(f"Processing Participant: {par}")

    # create a dictionary entry for the participant
    free_usage_dataset[par] = {}

    # loop over all datasets of the participant
    for num, dset in enumerate(raw_data[par]["freeMouse"]):

        # only select non-duplicate trials
        if dset not in duplicate_ids:

            print(f"Dset: {dset}")

            # start a timer to measure how long it takes to process one dataset
            # start = timer()

            # create a dict entry for the measurement timepoint
            free_usage_dataset[par][num] = {}

            # get task meta-information
            free_usage_dataset[par][num]["timestamp"] = raw_data[par]["freeMouse"][dset]["time"]
            free_usage_dataset[par][num]["zoom"] = raw_data[par]["freeMouse"][dset]["disInf"]["zoom"]
            free_usage_dataset[par][num]["screen_width"] = raw_data[par]["freeMouse"][dset]["disInf"]["screenSize"]["width"]
            free_usage_dataset[par][num]["screen_height"] = raw_data[par]["freeMouse"][dset]["disInf"]["screenSize"]["height"]

            # get the mouse data if it exists in the dataset (there are some cases in which no free mouse usage
            # datasets were saved), dont exclude the cases now, in order to inspect them in a later step
            if "mFreeUseData" in raw_data[par]["freeMouse"][dset]:
                free_mouse_data = raw_data[par]["freeMouse"][dset]["mFreeUseData"]

                # clean the dataset
                clean_data, artifacts, lockscreen_vals = clean_free_mouse_data(free_mouse_data)

                # save the data quality info of the measurement timepoint
                free_usage_dataset[par][num]["artifact_percent"] = artifacts
                free_usage_dataset[par][num]["total_datapoints"] = len(clean_data)
                free_usage_dataset[par][num].update(lockscreen_vals)

                # get data quality information regarding the sampling frequency
                free_usage_dataset[par][num].update(logging_interval_info(clean_data))

                # Split the data into movement episodes based on different pause threshold values
                # Because there is no "true" real pause threshold value, we create 3 different movement episode
                # datasets for three different pause threshold values of 1 second, 2 seconds and 3 seconds of no mouse
                # movement for a valid pause
                move_episodes_dsets = get_movement_episodes(clean_data, pause_thresholds=[1000, 2000, 3000])

                # calculate all mouse features for each movement episodes dataset
                for ep_dset in move_episodes_dsets:
                    # calculate the features
                    all_mouse_features = calculate_mouse_parameters_pipeline(move_episodes_dsets[ep_dset])
                    # add the features to the free usage dataset with the pause threshold value
                    for feat in all_mouse_features:
                        free_usage_dataset[par][num][ep_dset + "_" + feat] = all_mouse_features[feat]

            # get the arousal and valence values of the task
            valence = raw_data[par]["freeMouse"][dset]["selfReportData"]["valence"]
            arousal = raw_data[par]["freeMouse"][dset]["selfReportData"]["arousal"]
            free_usage_dataset[par][num]["valence"] = valence
            free_usage_dataset[par][num]["arousal"] = arousal
            # calculate the stress variable (if valence and arousal values are < 50)
            free_usage_dataset[par][num]["stress"] = 1 if arousal > 50 and valence < 50 else 0

            # get the sociodemographics of the participant
            free_usage_dataset[par][num]["age"] = raw_data[par]["Sociodem"]["age"]
            free_usage_dataset[par][num]["sex"] = raw_data[par]["Sociodem"]["sex"]
            free_usage_dataset[par][num]["hand"] = raw_data[par]["Sociodem"]["hand"]
            # save the participant ID
            free_usage_dataset[par][num]["ID"] = par
            # add the dataset ID (for data quality analysis in order to easily extract the raw dataset of the measurment)
            free_usage_dataset[par][num]["dset_ID"] = dset

            # print the time it took to process the dataset
            # print(f"Elapsed time to process the dataset: {timer() - start}")
print("All participants processed")


# convert the dictionary into a dataframe that has a row for each measurement timepoint
free_mouse_df = pd.concat({k: pd.DataFrame(v).T for k, v in free_usage_dataset.items()}, axis=0).round(4).reset_index(drop=True)


#%%

print(f"Shape of the entire features dataframe without duplicate trials: {free_mouse_df.shape}")
# save the dataset for data quality analysis
free_mouse_df.to_csv("Free_Mouse_Features_Data_Quality_Inspection.csv", index=False)

#%%

# if already saved, the data can also be imported
# free_mouse_df = pd.read_csv("Free_Mouse_Features_Data_Quality_Inspection.csv")

#%%

#########################
# Removal of bad trials #
#########################

# Duplicated trials are already removed, so they do not need to be handled here anymore.
# The removal procedure for the free mouse usage data also tries to eliminate bad trials

# There are some trials that have no mouse usage parameters (contains nan values), either because there was no
# recorded free mouse usage data for that trial in the dataset, or because the free mouse usage data had no mouse
# movement. These cases are excluded from the analysis dataset
cleaned_free_mouse_df = free_mouse_df.dropna()
print(f"Number of trials without recorded free mouse data or without recorded mouse movement: "
      f"{len(free_mouse_df) - len(cleaned_free_mouse_df)}")
print(f"Number of trials with mouse movement data: {len(cleaned_free_mouse_df)}")

#%%

# remove participants with too little trials (equal or less than 3 trials)
# add the number of timepoints per participant to the dataframe
cleaned_free_mouse_df["freq"] = cleaned_free_mouse_df.groupby("ID")["ID"].transform("count")
# get participants with low number of participations
low_participation = cleaned_free_mouse_df.loc[cleaned_free_mouse_df["freq"] <= 3]
print(f"Number of participants with 3 or less datasets: {low_participation['ID'].nunique()}")
print(f"Number of datasets from participants with 3 or less datasets: {len(low_participation)}")
cleaned_free_mouse_df = cleaned_free_mouse_df.loc[cleaned_free_mouse_df["freq"] > 3]
# compare their stress level with the stress level of the sample with more measurement timepoints
print(f"Number of Stressful measurements compared to the total sample for the low participation dataset "
      f"{len(low_participation.loc[low_participation['stress'] == 1]) / len(low_participation) * 100}%")
print(f"Number of Stressful measurements compared to the total sample for the cleaned dataset "
      f"{len(cleaned_free_mouse_df.loc[cleaned_free_mouse_df['stress'] == 1]) / len(cleaned_free_mouse_df) * 100}%")


print(f"Shape of dataset after removal of participants with too little trials: {cleaned_free_mouse_df.shape}")

#%%

# save the dataset as a csv file for data analysis
cleaned_free_mouse_df.to_csv("Free_Mouse_Features.csv", index=False)


#%%

############################################################################
# Get sample participant data and run the analysis with the sample dataset #
############################################################################

# also times the runtimes of the data preprocessing/calculation processes

# get a specific dataset
sample_data = raw_data['MDWbmhacEo']["freeMouse"]['-MnxqPR1NZrby6dsXMSQ']["mFreeUseData"]

# clean the dataset
start = timer()
clean_samp_data, art, locks = clean_free_mouse_data(sample_data)
print(f"Time for cleaning: {timer() - start}")

# get the movement episodes datasets
start = timer()
move_episodes_samples = get_movement_episodes(clean_samp_data, pause_thresholds=[1000, 2000])
print(f"Time for episode isolation: {timer() - start}")

# calculate the mouse usage parameters for the sample movement episodes
start = timer()
for i in move_episodes_samples:
    print(i)
    print(calculate_mouse_parameters_pipeline(move_episodes_samples[i]))
print(f"Time for mouse params calculation: {timer() - start}")
