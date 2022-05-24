'''
Code to evaluate the data quality of the tracked mouse free-mouse data in order to be able to make educated guesses
about data removal procedures/rules of potential outliers/bad cases. The data quality is inspected with rules-of-thumbs.
The mouse movement data is visually inspected for selected potential bad cases (univariate outliers of selected
variables)
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''


import numpy as np
import pandas as pd
import json
import gzip
import matplotlib.pyplot as plt

#%%

# import the raw dataset
with gzip.open("Datasets_Raw/FreeMouse_dataset.json.gz", "rb") as f:
    raw_data = json.loads(f.read())

# import the features dataset
data_quality_features = pd.read_csv("Free_Mouse_Features_Data_Quality_Inspection.csv")

#%%

# inspect the cases without mouse data
nan_data = data_quality_features[data_quality_features.isnull().any(axis=1)]

print(f"Number of trials without any movement data: {len(nan_data)}")
# There are two cases of no movement data
# Case 1: No free mouse usage data was saved (likely a data saving error/bug in the Study-App)
print(f"Number of trials without saved data: {nan_data['1000_movement_episodes'].isna().sum()}")
# Case 2: No mouse movement episode was extracted (at least 3 movement datapoints between pauses)
print(f"Number of trials without movement episodes: {len(nan_data) - nan_data['1000_movement_episodes'].isna().sum()}")

# The 9 Trials without saved data should be removed
# The 8 trials without movement data, but recorded data could potentially be included in the analysis. However,
# the goal of the study is to understand the relationship between emotional states and mouse usage, which cannot
# be evaluated if there is no mouse usage. Therefore, also remove this data

data_quality_features = data_quality_features.dropna()

#%%

# helper functions to process the raw data for plotting (copied from the feature calculation scripts)
# --------------------------------------------------------------------------------------------------


# use the helper function from the feature calculation procedure, but strip some calculations to just return the
# cleaned data
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

    # replace the lockscreen values with a previous valid value or 0 if there is no previous valid value (if mouse
    # usage recording starts with lockscreen values)
    mouse_df = mouse_df.mask(mouse_df.lt(-10000)).ffill().fillna(0)

    return mouse_df


# movement episode getter for one specific pause threshold
def get_movement_episodes(mouse_data, pause_threshold):

    # get the difference values between the mouse usage coordinates
    diff_data = mouse_data.diff()
    # select movement episodes = episodes with changes in x- & y-values
    diff_data["mo"] = np.where((diff_data["x"] != 0) | (diff_data["y"] != 0), 1, 0)

    # number pauses (all episodes with no movement change)
    numbered_pauses = []
    pause_num = 0
    pause_started = False
    for i in diff_data['mo']:
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
            pause_started = False
            numbered_pauses.append(0)

    # add the numbered pause list to the difference data
    diff_data["num_pauses"] = numbered_pauses

    # get the length of all marked pauses
    pause_length = diff_data.groupby(["num_pauses"]).sum()['t'][1:]

    # select "valid" pauses (no movement for a time that is longer than a specified threshold)
    # get the valid pauses for every specified pause threshold
    valid_pauses = pause_length.index[pause_length > pause_threshold].values

    # count number of movements between valid pauses
    # if smaller than 3, it was no movement, else, it was movement
    # label all valid movement episodes between valid pauses (valid movement episodes require at least 3 datapoints
    # with changes in the mouse cursor position)
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
        if i not in valid_pauses:
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

    return mouse_data_episode


#%%


# helper functions for the data quality analysis
# ----------------------------------------------

# helper to visualize the mouse movement
def visualize_mouse_movement(dframe, title):

    # simple scatterplot of the raw mouse usage data with connected consecutive mouse movement datapoints. The
    # data has unique colors per movement episode

    # get the number of movement episodes
    move_episodes = dframe["mo_episodes"].max()

    print(f"x: {dframe['x'].max()}, {dframe['x'].min()}")
    print(f"y: {dframe['y'].max()}, {dframe['y'].min()}")

    # setup a colormap
    # plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, move_episodes))))
    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    colors = [colormap(i) for i in np.linspace(0, 1, move_episodes)]

    # loop over all movement episodes
    for i in range(1, move_episodes + 1):
        # get the movement episode data
        mo_episode_data = dframe.loc[dframe["mo_episodes"] == i]
        # create a scatterplot of the movement data
        # linestyle='--', linewidth=0.8, alpha=0.6 # alternative option for potentially clearer visualization of many
        # movements
        plt.plot(mo_episode_data["x"], mo_episode_data["y"], color=colors[i-1])
        plt.scatter(x=mo_episode_data["x"], y=mo_episode_data["y"], color=colors[i-1], marker=".")

    # set a title
    plt.title(title)
    # hide the axis
    plt.axis("off")
    # show the plot
    plt.show()


# helper function to sort the feature dataset by columns and assess the data quality by inspecting the visualized data
def manually_inspect_data_quality(dframe, target_variable, pause_thresh, ascend=True):

    # sort the dataframe by the target column in the specified order
    sorted_df = dframe.sort_values(by=[target_variable], ascending=ascend)

    # loop over the rows of the sorted dataframe to be able to select the correct raw data for visualization
    # note that looping dataframe rows usually is an anti-pattern and should be avoided
    for index, row in sorted_df.iterrows():
        if input("Do you want to inspect the next case? [y/n]") == "y":

            # get the raw data of the participant
            par_raw_data = raw_data[row["ID"]]["freeMouse"][row["dset_ID"]]["mFreeUseData"]
            # clean the data
            clean_data = clean_free_mouse_data(par_raw_data)
            # extract the movement data based on a specified pause threshold
            move_data = get_movement_episodes(clean_data, pause_threshold=pause_thresh)
            # visualize the cleaned raw data
            print(f"Dset Value for {target_variable}: {row[target_variable]} for {row['ID']} and {row['dset_ID']}")
            visualize_mouse_movement(move_data, target_variable)
        else:
            print("Stopped the manual data inspection")
            break


#%%

#############################################################################
# Visually Expect potentially bad datasets based on selected mouse features #
#############################################################################

manually_inspect_data_quality(data_quality_features, "median_log_int", pause_thresh=1000, ascend=False)
# There is no clear indication of bad logging for high median loggin intervals when considering all logging intervals
# for the entire recorded mouse dataset

#%%

manually_inspect_data_quality(data_quality_features, "1000_recording_duration", pause_thresh=1000, ascend=True)
# for very short durations, there is not much mouse data. However, cases seem to be recorded with okay quality. There
# was one case with jumping mouse positions (the cause is unclear, and it might be due to a recording error, but does
# not have to be)

#%%

manually_inspect_data_quality(data_quality_features, "1000_movement_distance", pause_thresh=1000, ascend=False)
# Data seems to be fine, trials with long distances have a lot of mouse usage data

#%%

manually_inspect_data_quality(data_quality_features, "1000_mo_ep_mean_median_log_int", pause_thresh=1000, ascend=False)
# a high median logging interval indicates that the mouse logging did not work as expected. However, in contrast to
# the mouse usage task, high logging values do not indicate that all logged data is useless. Therefore, no data
# will be removed (the free mouse logging data is left untouched as compared to the mouse task data to account for the
# much lower standardization). An alternative would be to remove trials with a high logging interval. However, this
# would also the mouse usage features to change in a potentially undesired way

#%%

manually_inspect_data_quality(data_quality_features, "1000_mo_ep_mean_speed_mean", pause_thresh=1000, ascend=False)
# most cases look unsuspicious, there are some interesting cases for which it is unclear weather the mouse usage
# represents data collection errors or worked as intended and the participant had unusual mouse behavior

#%%

# Inspect the task num raw data to create the sample image for participant mouse usage behavior in task 16
# (the final image in the paper is flipped and the image size is set to figure(figsize=(6, 6)))
manually_inspect_data_quality(data_quality_features, "zoom", pause_thresh=1000, ascend=True)

#%%

########################################################
# Careful conclusion of the visual data quality analysis
#########################################################
# Because there is no "optimal movement" as opposed to the mouse usage task, which would indicate the data quality
# more clearly, data should not be removed. A better strategy might be to transform the data values to fit better
# to a normal distribution and force potential outliers closer to the mean. A manual outlier deletion process does
# not make sense and it is hard to justify an automatic outlier deletion process with specified thresholds

#%%

# Inspect the data of selected cases from the manual data inspection

# get a specific dataset
sample_data = raw_data['vrViYJPLXhR9zOmQkDopQIZIyCc2']["freeMouse"]['-MfZL2mdJFBWOnFvKYd9']["mFreeUseData"]

# clean the data
clean_samp_data = clean_free_mouse_data(sample_data)

# get the movement episodes
sample_episodes = get_movement_episodes(clean_samp_data, pause_threshold=1000)