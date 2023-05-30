'''
Code to get some descriptive statistics about the dataset (e.g. sociodemographics) as well as about the mouse-usage
features.
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''

#%%
# package imports
import gzip
import json
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import scale
import statsmodels.api as sm
from scipy.stats import norm

#%%

# dataset imports

# sociodemographic data of all participants
with gzip.open("C:/Users/Paul/Desktop/Data_Analysis/General_Analysis/sociodem_dataset.json.gz", "rb") as f:
     sociodem_data = json.loads(f.read())

# convert the dictionary into a pandas dataframe
sociodem_data = pd.DataFrame(sociodem_data).T

# mousetask data (cleaned)
mousetask_data = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis/Mouse_Task_Features.csv")

# import the processed free mouse usage features
free_mouse_data = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Free-Mouse_Analysis/Free_Mouse_Features.csv")


#%%

#####################################
# Get Basic Infos about the samples #
#####################################

# helper function to get basic descriptive stats about the sample
def sample_descriptive(sample):

    print(f"Total number of participants: {len(sample)}")
    print("\n")

    # age (1 = younger than 30, 2 = 30-39, 3 = 40-49, 4 = 50-59, 5 = 60 or older, -99 = missing)
    print(f"Age Distribution of the sample")
    for age, size in sample["age"].value_counts().iteritems():
        print(f"{size} participants in age group {age}")

    print("\n")
    # sex (0 = female, 1 = male, 2 = other, -99 = missing)
    print(f"Gender Distribution of the sample")
    for gend, size in sample["sex"].value_counts().iteritems():
        print(f"{size} participants of sex {gend}")

    print("\n")
    # hand that is used to control the computer mouse (0 = right, 1 = left)
    print(f"Mouse Usage Hand Distribution of the sample")
    for hand, size in sample["hand"].value_counts().iteritems():
        print(f"{size} participants use hand {hand} to navigate the computer mouse")

    print("\n")
    # os distribution (na values are filled with windows, because in an older version of the app, only mac os was
    # logged as the operating system)
    print(f"OS Distribution")
    for os, size in sample["os"].replace(to_replace='na', value='win32').value_counts().iteritems():
        print(f"{size} participants use os: {os}")

    print("\n")
    # sample distribution (recruited via the convenience sample versus via WisoPanel)
    print(f"Recruitment Distribution")
    for samp, size in sample["sample"].value_counts().iteritems():
        print(f"{size} participants recruited via: {samp}")

    print("\n")
    # more detailed sample info: Get info about participant renumeration of the WisoPanel Sample (plus the recruitment
    # time: the recruitment happened in waves, which was logged in the app version)
    # (1.0 = convenience sample first wave, 1.1 = convencience sample second wave, Panel_Pilot = Panel first wave,
    # Panel Fup = Panel second wave, Econd = financial participation renumeration in Euros
    print(f"Recruitment Details")
    for rec_det, size in sample["appVersion"].value_counts().iteritems():
        print(f"{size} used Study-App Version: {rec_det}")


# helper function to plot the valence/arousal distribution of the data
def plot_valence_arousal(data, filename):
    fig, ax = plt.subplots()
    # only show the upper and left spine
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    # draw a scatterplot of the relationship between valence and arousal
    val_arousal_plot = sns.scatterplot(data=data, x='valence', y='arousal', s=7)
    # add vertical and horizontal line
    val_arousal_plot.axhline(50, ls='--', color='black', alpha=0.6)
    val_arousal_plot.axvline(50, ls='--', color='black', alpha=0.6)
    # add custom x and y ticks
    val_arousal_plot.set(xticks=np.arange(0, 101, 50), yticks=np.arange(0, 101, 50))
    # add custom text to the axis
    ax.text(-0.05, 0.25, 'calm',
            horizontalalignment='center',
            verticalalignment='center',
            rotation='vertical',
            transform=ax.transAxes,
            fontsize=11)
    ax.text(-0.05, 0.75, 'excited',
            horizontalalignment='center',
            verticalalignment='center',
            rotation='vertical',
            transform=ax.transAxes,
            fontsize=11)
    ax.text(0.25, -0.05, 'negative',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=11)
    ax.text(0.75, -0.05, 'positive',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=11)

    # plt.show()
    plt.savefig(filename + '.png')
    plt.show()


#%%

# Info about the entire sample
# -----------------------------
# (all participants that downloaded the Study-App and started the study)

print("Dem infos about the entire dataset/all participants and trials\n")
sample_descriptive(sociodem_data)
# Info about the Number of Datasets are calculated in the mouse_Task_feature_calculation file, because duplicated
# datasets needed to be removed, which requires the raw data


#%%

# Descriptive Stats about Valence and Arousal Across both, the mouse task dataset and the free-mouse dataset
# merge both datasets into one

# create subsets of the mouse datasets only containing valence, arousal and the dset id
mouse_data_subset = mousetask_data[['dset_ID', 'arousal', 'valence', 'ID']]
free_data_subset = free_mouse_data[['dset_ID', 'arousal', 'valence', 'ID']]

# merge the data subsets to create a dataset that contains all "valid" arousal and valence data that was used in the study
merged_val_arousal_dset = pd.merge(mouse_data_subset, free_data_subset, on='dset_ID', how='outer')
merged_val_arousal_dset['arousal'] = merged_val_arousal_dset['arousal_x'].combine_first(merged_val_arousal_dset['arousal_y'])
merged_val_arousal_dset['valence'] = merged_val_arousal_dset['valence_x'].combine_first(merged_val_arousal_dset['valence_y'])
merged_val_arousal_dset['ID'] = merged_val_arousal_dset['ID_x'].combine_first(merged_val_arousal_dset['ID_y'])

# Get descriptive stats about valence, arousal & stress in the mouse task dataset
print(f"Descriptive stats about valence:\n{merged_val_arousal_dset['valence'].describe()}\n")
print(f"Descriptive stats about arousal:\n{merged_val_arousal_dset['arousal'].describe()}\n")

# get descriptive stats about the valence and arousal distribution per participant
print(f"Descriptive stats about valence per participant:\n{merged_val_arousal_dset.groupby('ID')['valence'].describe()}\n")
print(f"Average standard deviation in valence per participant: {merged_val_arousal_dset.groupby('ID')['valence'].std().mean()}\n")
print(f"Descriptive stats about arousal per participant:\n{merged_val_arousal_dset.groupby('ID')['arousal'].describe()}\n")
print(f"Average standard deviation in arousal per participant: {merged_val_arousal_dset.groupby('ID')['arousal'].std().mean()}")

# plot valence and arousal
plot_valence_arousal(merged_val_arousal_dset, 'task_valence_arousal_plot')

#%%

# Info about the mouse task dataset
# ---------------------------------

# Dropped participants/trials because:
# - mouse usage recording errors
# - too few trials

# drop all participants which are not in the final mouse task sample anymore
# Participants are dropped because they completed 3 or less data collections (4 participants completed no data
# data collection and just finished the app tutorial)
mouse_task_index = mousetask_data["ID"].unique()
mouse_task_dem = sociodem_data[sociodem_data.index.isin(mouse_task_index)]
print("Dem infos about the mouse task dataset\n")
sample_descriptive(mouse_task_dem)

# Get the total number of collected datasets from the mouse task sample
print(f"\nTotal number of individual data measurements in the mouse task sample: {len(mousetask_data)}\n")

# descriptive stats about the collected datasets per participant
dsets_per_par = mousetask_data.groupby("ID")["freq"].unique().explode()
print(f"Descriptive Stats about the number of data measurements per participant in the mouse task sample:"
      f"\n{pd.to_numeric(dsets_per_par).describe()}\n")

# get the median of the median of the median sampling frequency
print(f"Median Sampling Frequency: {np.median(mousetask_data['median_sampling_freq'])}\n")
# get the median task time
print(f"Median Task Duration: {np.median(mousetask_data['task_duration'])}\n")

# Get descriptive stats about valence, arousal & stress in the mouse task dataset
print(f"Descriptive stats about valence in the mouse task sample:\n{mousetask_data['valence'].describe()}\n")
print(f"Descriptive stats about arousal in the mouse task sample:\n{mousetask_data['arousal'].describe()}\n")
# 0 = No-stress, 1 = stress
print(f"Descriptive stats about stress in the mouse task sample:\n{mousetask_data['stress'].value_counts()}")

# get descriptive stats about the valence and arousal distribution per participant
print(f"Descriptive stats about valence per participant in the mouse task sample:\n{mousetask_data.groupby('ID')['valence'].describe()}\n")
print(f"Average standard deviation in valence per participant: {mousetask_data.groupby('ID')['valence'].std().mean()}\n")
print(f"Descriptive stats about arousal per participant in the mouse task sample:\n{mousetask_data.groupby('ID')['arousal'].describe()}\n")
print(f"Average standard deviation in arousal per participant: {mousetask_data.groupby('ID')['arousal'].std().mean()}")

# plot it
plot_valence_arousal(mousetask_data, 'task_valence_arousal_plot')

#%%

# Info about the free mouse dataset
# ---------------------------------

# Dropped participants/trials because:
# - no recorded movement data

# repeat the same as above, but with the free mouse data
free_mouse_index = free_mouse_data["ID"].unique()
free_mouse_dem = sociodem_data[sociodem_data.index.isin(free_mouse_index)]
print("Dem infos about the free mouse usage dataset\n")
sample_descriptive(free_mouse_dem)

# Get the total number of collected datasets from the mouse task sample
print(f"\nTotal number of individual data measurements in the free mouse usage dataset: {len(free_mouse_data)}\n")

# descriptive stats about the collected datasets per participant
dsets_per_par = free_mouse_data.groupby("ID")["freq"].unique().explode()
print(f"Descriptive Stats about the number of data measurements per participant in the free mouse usage dataset:"
      f"\n{pd.to_numeric(dsets_per_par).describe()}\n")

# Get descriptive stats about valence, arousal & stress in the mouse task dataset
print(f"Descriptive stats about valence in the free mouse usage dataset:\n{free_mouse_data['valence'].describe()}\n")
print(f"Descriptive stats about arousal in the free mouse usage dataset:\n{free_mouse_data['arousal'].describe()}\n")
# 0 = No-stress, 1 = stress
print(f"Descriptive stats about stress in the free mouse usage dataset:\n{free_mouse_data['stress'].value_counts()}")

# get descriptive stats about the valence and arousal distribution per participant
print(f"Descriptive stats about valence per participant in the free mouse usage dataset:\n{free_mouse_data.groupby('ID')['valence'].describe()}\n")
print(f"Average standard deviation in valence per participant: {free_mouse_data.groupby('ID')['valence'].std().mean()}\n")
print(f"Descriptive stats about arousal per participant in the free mouse usage dataset:\n{free_mouse_data.groupby('ID')['arousal'].describe()}\n")
print(f"Average standard deviation in arousal per participant: {free_mouse_data.groupby('ID')['arousal'].std().mean()}")

# plot the valence/arousal distribution
plot_valence_arousal(free_mouse_data, 'free_mouse_valence_arousal_plot')


#%%

###################################################################################################
# Get descriptive stats about the mouse usage features in the task dataset and free mouse dataset #
###################################################################################################

# have a rename dictionary to give all features "prettier names" in the plots
# THIS IS NOT A GOOD CODING SOLUTION
rename_dict = {
    # task features
    "task_duration": 'Task: Duration',
    "clicks": 'Clicks',
    "task_total_dist": "Task: Tot. Distance",
    'task_speed_mean': 'Task: Speed (mean)',
    'task_speed_sd': 'Task: Speed (sd)',
    'task_abs_accel_mean': 'Task: Accel (mean)',
    'task_abs_accel_sd': 'Task: Accel (sd)',
    'task_abs_jerk_mean': 'Task: Jerk (mean)',
    "task_abs_jerk_sd": 'Task: Jerk (sd)',
    'task_angle_mean': 'Task: Angle (mean)',
    "task_angle_sd": 'Task: Angle (sd)',
    "task_x_flips": 'Task: X-Flips',
    "task_y_flips": 'Task: Y-Flips',
    "trial_mean_duration": 'Trial (mean): Duration',
    "trial_sd_duration": 'Trial (sd): Duration',
    "trial_mean_trial_move_offset": 'Trial (mean): Initiation Time',
    "trial_sd_trial_move_offset": 'Trial (sd): Initiation Time',
    "trial_sd_total_dist": 'Trial (sd): Tot. Distance',
    "trial_sd_distance_overshoot": 'Trial (sd): Ideal Line Deviation',
    'trial_mean_distance_overshoot': 'Trial (Mean): Ideal Line Deviation',
    "trial_mean_speed_mean": 'Trial (mean): Speed (mean)',
    "trial_sd_speed_mean": 'Trial (sd): Speed (mean)',
    "trial_mean_speed_sd": 'Trial (mean): Speed (sd)',
    "trial_sd_speed_sd": 'Trial (sd): Speed (sd)',
    "trial_sd_abs_jerk_sd": 'Trial (sd): Jerk (sd)',
    "trial_mean_angle_mean": 'Trial (mean): Angle (mean)',
    "trial_sd_angle_mean": 'Trial (sd): Angle (mean)',
    "trial_mean_angle_sd": 'Trial (mean): Angle (sd)',
    "trial_sd_angle_sd": 'Trial (sd): Angle (sd)',
    "trial_mean_x_flips": 'Trial (mean): X-Flips',
    "trial_mean_y_flips": 'Trial (mean): Y-Flips',
    "trial_sd_x_flips": 'Trial (sd): X-Flips',
    "trial_sd_y_flips": 'Trial (sd): Y-Flips',
    "trial_mean_total_dist": 'Trial (mean): Tot. Distance',
    "trial_sd_abs_jerk_mean": 'Trial (sd): Jerk (mean)',
    "trial_mean_abs_jerk_mean": 'Trial (mean): Jerk (mean)',
    "trial_mean_abs_jerk_sd": 'Trial (mean): Jerk (sd)',
    'trial_mean_abs_accel_mean': 'Trial (mean): Accel (mean)',
    'trial_sd_abs_accel_mean': 'Trial (sd): Accel (mean)',
    'trial_mean_abs_accel_sd': 'Trial (mean): Accel (sd)',
    'trial_sd_abs_accel_sd': 'Trial (sd): Accel (sd)',
    # free mouse features
    "recording_duration": 'Recording Duration',
    "mo_ep_mean_episode_duration": 'Move Ep. (mean): Episode Duration',
    "mo_ep_mean_total_dist": 'Move Ep. (mean): Tot. Distance',
    "mo_ep_mean_speed_mean": 'Move Ep. (mean): Speed (mean)',
    "mo_ep_sd_speed_mean": 'Move Ep. (sd): Speed (mean)',
    "mo_ep_mean_angle_mean": 'Move Ep. (mean): Angle (mean)',
    "mo_ep_sd_angle_sd": 'Move Ep. (sd): Angle (sd)',
    "mo_ep_mean_x_flips": 'Move Ep. (mean): X-Flips',
    "no_movement": 'No Movement',
    "lockscreen_time": 'Lockscreen Time',
    "movement_episodes": 'Num. of Move Ep.',
    "mo_ep_sd_abs_jerk_mean": 'Move Ep. (sd): Jerk (mean)',
    "mo_ep_sd_angle_mean": 'Move Ep. (sd): Angle (mean)',
    "mo_ep_mean_angle_sd": 'Move Ep. (mean): Angle (sd)',
    "mo_ep_mean_speed_sd": 'Move Ep. (mean): Speed (sd)',
    "mo_ep_sd_total_dist": 'Move Ep. (sd): Tot. Distance',
    "mo_ep_mean_y_flips": 'Move Ep. (mean): Y-Flips',
    'mo_ep_sd_episode_duration': 'Move Ep. (sd): Episode Duration',
    'mo_ep_sd_speed_sd': 'Move Ep. (sd): Speed (sd)',
    'mo_ep_mean_abs_accel_mean': 'Move Ep. (mean): Accel (mean)',
    'mo_ep_sd_abs_accel_mean': 'Move Ep. (sd): Accel (mean)',
    'mo_ep_mean_abs_accel_sd': 'Move Ep. (mean): Accel (sd)',
    'mo_ep_sd_abs_accel_sd': 'Move Ep. (sd): Accel (sd)',
    'mo_ep_mean_abs_jerk_mean': 'Move Ep. (mean): Jerk (mean)',
    'mo_ep_mean_abs_jerk_sd': 'Move Ep. (mean): Jerk (sd)',
    'mo_ep_sd_abs_jerk_sd': 'Move Ep. (sd): Jerk (sd)',
    'mo_ep_sd_x_flips': 'Move Ep. (sd): X-Flips',
    'mo_ep_sd_y_flips': 'Move Ep. (sd): Y-Flips',
    'movement_duration': 'Movement Duration',
    'movement_distance': 'Total Distance',
    'lockscreen_episodes:': 'Num. of Lockscreen Ep.',
    # valence and arousal
    'valence': ' Valence',
    'arousal': 'Arousal',
    # Sociodemographic Variables
    'zoom': 'Zoom',
    'screen_width': ' Screen Width',
    'screen_height': ' Screen Height',
    'age': "Age",
    'sex': 'Sex',
    'hand': 'Hand'
}


# helper function to create an Order colummn in the dataset that indicates the number of Measure in the study
# (i.e. the order variable shows the i th time the participant took part in the data collection)
def add_order(x):
    x = x.sort_values(by=["timestamp"])
    x['Order'] = range(len(x))
    return x


# helper function to create a kdeplot of selected mouse usage features
def multi_kde_plot(data, name):

    # set a style
    sns.set_style("white")

    # first rename the columns of the dataframe to give the variables in the plots "prettier" names
    # this is not an ideal coding solution!
    data.columns = [rename_dict.get(x, x) for x in data.columns]

    # create a plot with an appropriate number of columns and rows (depending of the number of the columns to plot
    num_cols = data.shape[1]

    fig, axes = plt.subplots(nrows=int(np.sqrt(num_cols)) + 1, ncols=int(np.sqrt(num_cols)) + 1,
                             figsize=(30, 30), sharex=False, sharey=False)
    axes = axes.ravel()  # array to 1D
    cols = list(data.columns)  # create a list of dataframe columns to use

    for col, ax in zip(cols, axes):
        sns.set(font_scale=2.25)
        sns.kdeplot(data=data, x=col, fill=True, ax=ax)
        ax.set(title=col, xlabel=None, xticklabels=[], yticklabels=[])

    # delete the empty subplots
    ax_to_del = [i for i in range(num_cols, len(axes))]

    for i in ax_to_del:
        fig.delaxes(axes[i])

    fig.tight_layout()
    plt.savefig(name + '_kde_plot.png')
    plt.show()


# helper function to plot a correlation heatmap
def correlation_heatmap(data, fig_size, font_scale, name, add_text=True):

    # first rename the columns of the dataframe to give the variables in the plots "prettier" names
    # this is not an ideal coding solution!
    data.columns = [rename_dict.get(x, x) for x in data.columns]

    # calculate the correlation matrix of the data
    corr = data.corr()

    # set a figure size
    plt.figure(figsize=fig_size)
    # set a scale size to scale all texts
    sns.set(font_scale=font_scale)
    sns.set_style("white")
    # create a mask to only plot one diagonal of the heatmap
    mask = np.tril(np.ones(corr.corr().shape)).astype(bool)
    # get the correlation matrix
    corr = corr.where(mask)
    # create the heatmap
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        linewidth=0.8,
        cmap=sns.diverging_palette(220, 20, n=200),
        fmt='.2f',
        cbar_kws={"shrink": .45, "label": 'correlation coeff.'},
        annot=add_text,
        square=True,
    )
    # set the axis labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
    )

    # if specified, add the correlation coefficient as text into the tiles
    if add_text:
        # change the text format to remove the 0 before the decimals and to replace 1 with an empty string
        for t in ax.texts: t.set_text(t.get_text().replace('0.', '.').replace('1.00', ''))

    # set a tight layout
    plt.tight_layout()
    plt.savefig(name + '_corr_heatmap.png')
    # plot the heatmap
    plt.show()


#  helper function to create a pairplot of the selected variables
# create a custom pairplot to visualize the relationship between the data
# adapted from: https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another
def plot_pairplot(data, name):

    # first rename the columns of the dataframe to give the variables in the plots "prettier" names
    # this is not an ideal coding solution!
    data.columns = [rename_dict.get(x, x) for x in data.columns]

    sns.set(font_scale=1.7, style="darkgrid")

    # setup a custom function to plot the correlation coefficient
    def corr_coeff(x, y, **kwargs):
        ax = plt.gca()
        ax.set_axis_off()
        # calculate the correlation coefficient between the variables
        r = np.corrcoef(x, y)[0, 1]
        r_text = f"{r:2.2f}".replace('0.', '.')

        ax.scatter([.5], [.5], 100000, [r], cmap='coolwarm',
                   vmin=-1, vmax=1, transform=ax.transAxes, marker='s')

        ax.annotate(r_text, xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')

    # set up a pairgrid
    grid = sns.PairGrid(data, diag_sharey=False, aspect=1.4)

    # plot the scatterplot of the variables on the lower triangle
    grid.map_lower(sns.scatterplot, color='0.3', s=5)
    # grid.map_lower(sns.kdeplot, cmap="Blues_d") # alternative, less cluttered visualization of the bivariate
    # distribution
    # plot a histogram of each variable on the diag
    grid.map_diag(sns.kdeplot) # sns.histplot, kde=True
    # plot a correlation heatmap (custom function) on the upper triangle
    grid.map_upper(corr_coeff)

    for ax in grid.axes.flatten():
        # rotate x axis labels
        ax.set_xlabel(ax.get_xlabel(), rotation=30)
        # rotate y axis labels
        ax.set_ylabel(ax.get_ylabel(), rotation=0)
        # set y labels alignment
        ax.yaxis.get_label().set_horizontalalignment('right')

    # set a tight layout
    plt.tight_layout()
    # plt.savefig(name + '_pairplot.png')
    # show the plot
    plt.show()

# Helper function for the outlier removal process in the mouse task dataset
# the chosen outlier removal method is based on the interquartile range because it is robust against outliers
# (of course, other removal methods exist, e.g. MAD-outlier removal, and could be used, but testing "everything"
# is not feasible)
# this code should probably be improved
def remove_iqr_outliers(df, thresh):

    # calculate the interquantile range for all mouse usage features
    q1 = df[all_mouse_task_features].quantile(0.25)
    q3 = df[all_mouse_task_features].quantile(0.75)
    IQR = q3 - q1

    # remove all datapoints that fall outside of the IQR range * the specified threshold
    df = df[~((df[all_mouse_task_features] < (q1 - thresh * IQR)) |
              (df[all_mouse_task_features] > (q3 + thresh * IQR))).any(axis=1)]

    return df


# simple standardization helper function to standardize selected columns in groupby method
def standardize_cols(df, cols):

    df[cols] = df[cols].apply(lambda x: scale(x))

    return df


# helper function to perform rank-based inverse normal transformation on a column in a pandas dataframe
# see: https://agleontyev.netlify.app/post/rin_transform3/
def rank_inverse_normal_transform(ds):
    ds_rank = ds.rank()
    numerator = ds_rank - 0.5
    par = numerator/len(ds)
    result = norm.ppf(par)
    return result


# helper function that mimics the caret package findCorrelation with setting exact = F
# adapted from: https://stackoverflow.com/questions/41761332/translate-r-function-caretfindcorrelation-to-python-3-via-pandas-using-vectori
def find_corr_vars(data, cutoff=0.8):
    """
    search correlation matrix and identify pairs that if removed would reduce pair-wise correlations
        args:
            data: a dataframe
            cutoff: pairwise absolute correlation cutoff
        returns:
            a list of correlated variables that can be removed
    """

    # get the correlation matrix of the dataset
    corrmat = data.corr().abs()

    # get the average correlation of each column
    average_corr = corrmat.abs().mean(axis=1)

    # set lower triangle and diagonal of correlation matrix to NA
    corrmat = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool))

    # where a pairwise correlation is greater than the cutoff value, check whether mean abs.corr of a or b is greater
    # and cut it
    to_delete = list()
    for col in range(0, len(corrmat.columns)):
        for row in range(0, len(corrmat)):
            if corrmat.iloc[row, col] > cutoff:
                # print(f"Compare {corrmat.index.values[row]} with {corrmat.columns.values[col]}")
                if average_corr.iloc[row] > average_corr.iloc[col]:
                    to_delete.append(corrmat.index.values[row])
                else:
                    to_delete.append(corrmat.columns.values[col])

    to_delete = list(set(to_delete))

    return to_delete


# simple plot to visualize the relationship between the order (time) and a target variable for a randomly drawn
# subset of participants
def plot_habituation_multi_pars(data, target, name):

    # select a subset of the dataset to make the plot a little bit better to view (there are more than 150 participants
    # in the dataset, plotting them at once is too much

    # set a figure size
    # plt.figure(figsize=fig_size)
    # set a scale size to scale all texts
    sns.set(font_scale=1)
    sns.set_style("white")

    # specify the number of random IDs to draw
    N = 12
    # draw the random IDs and select the relevant data
    random_ids = np.random.choice(data['ID'].unique(), N, replace=False)
    selected_data = data[data['ID'].isin(random_ids)]

    # create a scatter plot with a (linear) trend line between the order and the target var for each selected
    # participant
    sns.lmplot(data=selected_data, x="Order", y=target, col="ID", height=3, facet_kws=dict(sharex=False, sharey=False))

    plt.savefig(name + "_" + target + '.png')
    plt.show()



# another simple plot to visualize the relationship between the order (time) and a target variable for a randomly drawn
# subset of participants. This plot draws a lineplot for each (randomly selected) participant in the same plot
def plot_habituation_single_pars(data, target, name):

    # set a scale size to scale all texts
    sns.set(font_scale=0.85)
    sns.set_style("white")

    # specificy the random number of participants that are drawn and plotted
    N = 12
    # draw the random IDs and select the relevant data
    random_ids = np.random.choice(data['ID'].unique(), N, replace=False)
    selected_data = data[data['ID'].isin(random_ids)]

    # create a simple lineplot for each participant (as indicated by the HUE) in a different color
    sns.lineplot(data=selected_data, x="Order", y=target, hue="ID", lw=0.8)
    # remove the legend
    plt.legend([], [], frameon=False)

    plt.savefig(name + "_" + target + '.png')
    plt.show()



# run individual regression of order (time) on a target variable for each participant to test the (linear) effect of
# order (time) on the target variable and plot the individual order effect coefficients and their confidence intervals
def plot_habituation_all_pars(data, target, name):
    # set a scale size to scale all texts
    sns.set(font_scale=1)
    sns.set_style("white")

    # helper function to run a (linear) regression per participant
    def _per_participant_regression(data, yvar, xvars):
        # get the DV and IV variables
        Y = data[yvar]
        X = data[xvars]
        # add an intercept to the model
        X['intercept'] = 1.
        # run the linear regression
        result = sm.OLS(Y, X).fit()
        # get the coefficient parameters and the confidence intervals
        params = result.params.loc["Order"]
        ci = result.conf_int(alpha=0.05, cols=None).loc["Order"]
        ci.index = ['conf_low', 'conf_high']
        # add both together and return them
        ci["estimate"] = params
        return ci

    # Get the by participant regression coefficients and confidence intervals
    participant_estimates = data.groupby('ID').apply(_per_participant_regression, target, ['Order'])
    participant_estimates.reset_index(drop=True, inplace=True)
    # sort the values by their size for better plotting
    participant_estimates = participant_estimates.sort_values(by=["estimate"])
    # add an ID variable
    participant_estimates["ID"] = range(len(participant_estimates))

    # create the plot

    # first, loop all participants and create a line of their confidence interval
    for index, row in participant_estimates.iterrows():
        plt.hlines(row["ID"], row["conf_low"], row["conf_high"], colors="blue", alpha=0.2)
    # next, create a scatterplot of the estimate
    sns.scatterplot(x="estimate", y="ID", data=participant_estimates, color="blue")
    # finally, add a vertical line that indicates 0
    plt.vlines(0, 0, len(participant_estimates), linestyles="dashed", colors="black")

    plt.savefig(name + "_" + target + '.png')
    plt.show()


#%%

# Mouse Task Descriptive Analysis
# -------------------------------

# get all calculated mouse usage features
all_mouse_task_features = ['task_duration', 'clicks', 'task_total_dist', 'task_speed_mean',
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

control_vars = ['zoom', 'screen_width', 'screen_height', 'age', 'sex', 'hand', 'Order']

# add the oder feature to the dataset
mousetask_data = mousetask_data.groupby("ID", group_keys=False).apply(func=add_order)
mousetask_data.reset_index(drop=True, inplace=True)


#%%

# in a first step. get a feeling for the distribution of the raw mouse usage features
multi_kde_plot(mousetask_data.loc[:, all_mouse_task_features + ["valence", "arousal"]], 'mouse_task_all')

# Most kde plots suggest a close to normal distribution for the mouse features, except the task time related features.
# However, there are 2 cases with extremly long task times, which might heavily skew the plots --> they should
# be removed, also because taking a long break during the mouse task is unwanted usage behavior

#%%

# Because there exist many data preprocessing options and to get a better sense for the results and their stability
# depending on data preprocessing, the dataset is processed in multiple ways and the analysis are performed on each
# dataset (here, we get the descriptive stats about each dataset)

# The preprocessing includes outlier removal (based on the mouse task), linear equating to adapt for differences in the
# mouse usage parameters due to different mouse usage tasks and removal of redundant features.
# To get a better sense for the stability of the results, we used three different outlier removal procedures and
# created three different preprocessed datasets that will be analyzed
# Note that this preprocessing routine is only one of many (and maybe potentially infinite) options. Other routines
# could use different steps or choose other options within each step

# initialize a dictionary that contains the entire dataset
mouse_task_datasets = {}

# set the outlier removal options: use the entire dataset (only remove few cases with a very large task time) or use
# the iqr_outlier removal procedure with a specified threshold
outlier_options = ["all", 2.5, 3.5]

# loop the outlier options to create the datasets
for opt in outlier_options:

    save_string = opt if opt == "all" else "iqr_" + str(opt)
    print(f"Processing the dataset with outlier option {save_string}")

    # if all are selected
    if opt == "all":
        # only remove datasets with a task duration greater than 5 minutes
        dset = mousetask_data.loc[mousetask_data["task_duration"] < 300]
    # if the outlier option is an interquantile outlier threshold
    else:
        # group the data by the mouse task and remove outliers by the task
        dset = mousetask_data.groupby('taskNum').apply(
            lambda x: remove_iqr_outliers(x, thresh=opt)).reset_index(drop=True)

    # now linear equate the different tasks to account for potential differences in the mouse usage params per task
    dset = dset.groupby('taskNum').apply(
        lambda x: standardize_cols(x, all_mouse_task_features)).reset_index(drop=True)

    # now remove highly correlated features to reduce the redundancies in the dataset
    redundant_features = find_corr_vars(dset.loc[:, all_mouse_task_features], cutoff=0.8)
    print(f"Number of removed collinear features: {len(redundant_features)}")
    print(f"Number of remaining mouse features: {len(all_mouse_task_features) - len(redundant_features)}")
    dset = dset.drop(redundant_features, axis=1)

    # save the dataset in the list
    mouse_task_datasets[save_string] = dset



#%%

# get some basic information about the datasets after applying the outlier procedures (we skip every other dataset
# because the information are the same independent of the standardization approach)
for n, dset in enumerate(mouse_task_datasets):
    print(f"Len of Dataset: {dset}: {len(mouse_task_datasets[dset])}")
    print(f"Descriptive stats about valence:\n{mouse_task_datasets[dset]['valence'].describe()}\n")
    print(f"Descriptive stats about arousal:\n{mouse_task_datasets[dset]['arousal'].describe()}\n")
    # 0 = No-stress, 1 = stress
    print(f"Descriptive stats about stress:\n{mouse_task_datasets[dset]['stress'].value_counts()}\n")
    print(f"Percentage of stress:\n"
          f"{mouse_task_datasets[dset]['stress'].value_counts()[1] / mouse_task_datasets[dset]['stress'].value_counts()[0] * 100}\n")
    # plot_valence_arousal(mouse_task_datasets[dset], dset)

#%%

# plot the kde plots for all datasets
for dset in mouse_task_datasets:
    print(f"KDE plots for: {dset}")
    # get the remaining task features
    remaining_features = [feat for feat in all_mouse_task_features if feat in list(mouse_task_datasets[dset].columns)]
    multi_kde_plot(mouse_task_datasets[dset].loc[:, remaining_features + ["valence", "arousal"]], dset)


#%%

# plot a correlation heatmap for all datasets
for dset in mouse_task_datasets:
    print(f"Correlation Heatmap for: {dset}")
    remaining_features = [feat for feat in all_mouse_task_features if feat in list(mouse_task_datasets[dset].columns)]
    correlation_heatmap(mouse_task_datasets[dset].loc[:, control_vars + remaining_features + ["valence", "arousal"]],
                        fig_size=(48,38), font_scale=3.2, name=dset, add_text=True)


#%%

# inspect the relationship between the order and outcome and predictor variables more closely to get a descriptive
# feeling for a potential habituation effect throughout the study

# Do this using data visualization

# doing this for all variables in all datasets is a little bit much, therefore, do it for (randomly) selected variables
# in a randomly selected dataset (of course, it is also possible to select a target and dataset manually)
rand_task_dset_name, rand_task_dset = random.choice(list(mouse_task_datasets.items()))
random_target = random.choice([feat for feat in all_mouse_task_features if feat in rand_task_dset.columns] +
                              ["valence", "arousal"])

#%%

# use the three different visualization approaches to descriptively check out the relationship between order and the
# target
plot_habituation_single_pars(rand_task_dset, random_target, rand_task_dset_name)

#%%
plot_habituation_multi_pars(rand_task_dset, random_target, rand_task_dset_name)

#%%
plot_habituation_all_pars(rand_task_dset, random_target, rand_task_dset_name)


#%%

# In the KDE-Plots it was noticable that not all mouse usage features as well as the dependent variables did now always
# follow a normal-distribution. One reviewer remarked that it might be better to transform the data before data analysis
# to make it more norma-like. We therefore applied rank-based inverse normal transformation to the data before feeding
# it into the mixed-level analysis. Here, use the transformation to check out how it changes the distribution (and
# possibly correlations) of the features

# loop the datasets to add the transformed features to each dataset
for dset in mouse_task_datasets:
    print(f"Transforming Features for Dataset: {dset}")
    # get the remaining features in the dataset
    remaining_features = [feat for feat in all_mouse_task_features if feat in list(mouse_task_datasets[dset].columns)]
    # loop all remaining features and add the transformed features to the dataset
    for feat in remaining_features + ["valence", "arousal"]:
        mouse_task_datasets[dset][feat + "_trans"] = rank_inverse_normal_transform(mouse_task_datasets[dset][feat])

#%%

# create a kde plot for the transformed feature in each dataset
for dset in mouse_task_datasets:
    print(f"KDE plots after feature transformation: {dset}")
    trans_features = [feat + "_trans" for feat in all_mouse_task_features if feat in list(mouse_task_datasets[dset].columns)]
    multi_kde_plot(mouse_task_datasets[dset].loc[:, trans_features + ["valence_trans", "arousal_trans"]], dset)

#%%

# create a correlation heatmap for the transformed features in each dataset
for dset in mouse_task_datasets:
    print(f"Heatmap after feature transformation: {dset}")
    trans_features = [feat + "_trans" for feat in all_mouse_task_features if
                      feat in list(mouse_task_datasets[dset].columns)]
    correlation_heatmap(mouse_task_datasets[dset].loc[:, control_vars + trans_features + ["valence_trans", "arousal_trans"]],
                        fig_size=(48, 38), font_scale=3.2, name=dset, add_text=True)



#%%

# Free Mouse Usage descriptive analysis
# -------------------------------------

# get the mouse usage features
free_mouse_feats = ['recording_duration', 'movement_episodes', 'mo_ep_mean_episode_duration',
                    'mo_ep_sd_episode_duration', 'mo_ep_mean_total_dist', 'mo_ep_sd_total_dist',
                    'mo_ep_mean_speed_mean', 'mo_ep_sd_speed_mean', 'mo_ep_mean_speed_sd', 'mo_ep_sd_speed_sd',
                    'mo_ep_mean_abs_accel_mean', 'mo_ep_sd_abs_accel_mean', 'mo_ep_mean_abs_accel_sd',
                    'mo_ep_sd_abs_accel_sd', 'mo_ep_mean_abs_jerk_mean', 'mo_ep_sd_abs_jerk_mean',
                    'mo_ep_mean_abs_jerk_sd', 'mo_ep_sd_abs_jerk_sd', 'mo_ep_mean_angle_mean',
                    'mo_ep_sd_angle_mean', 'mo_ep_mean_angle_sd', 'mo_ep_sd_angle_sd', 'mo_ep_mean_x_flips',
                    'mo_ep_sd_x_flips', 'mo_ep_mean_y_flips', 'mo_ep_sd_y_flips', 'movement_duration',
                    'movement_distance', 'no_movement', 'lockscreen_episodes:', "lockscreen_time"]

control_vars = ['zoom', 'screen_width', 'screen_height', 'age', 'sex', 'hand', 'Order']

#%%

# add the order variable to the free mouse features datase
free_mouse_data = free_mouse_data.groupby("ID", group_keys=False).apply(func=add_order)
free_mouse_data.reset_index(drop=True, inplace=True)


#%%

# similar to the mouse task, we create different datasets that will be analyzed because there exist different
# preprocessing options. However, the preprocessing options are different from the mouse usage task.
# Regarding free mouse usage, there is no "right" or "wrong" usage (such as not doing the task properly).
# Therefore, no outlier trials are removed. In the free mouse dataset we used different pause thresholds:
# - the pause threshold value that separates movement trials
# (- a threshold about the required ammount of datapoints/logging time for each 5 minute interval)

# create the free mouse usage datasets
free_mouse_datasets = {}

pause_threshs = ["1000", "2000", "3000"]

# loop the thresholds and create the datasets
for thresh in pause_threshs:

    print(f"Processing the Dataset with a Pause Threshold of: {thresh}")

    # get the features for the specified threshold
    dset = free_mouse_data.loc[:, [thresh + "_" + i if thresh + "_" + i in free_mouse_data.columns
                               else i for i in free_mouse_feats]]
    # rename the columns to exclude the thresh value
    dset.columns = free_mouse_feats

    # add all non mouse feature columns to the dset (concat the dataframe with a dataframe that only contains the
    # non feature columns, which are filtered out using a regular expression
    dset = pd.concat([pd.DataFrame(dset, columns=free_mouse_feats), free_mouse_data[free_mouse_data.columns.drop(
        list(free_mouse_data.filter(regex='|'.join(free_mouse_feats))))]], axis=1)

    # remove highly correlated mouse features from the dataset
    redundant_features = find_corr_vars(dset.loc[:, free_mouse_feats], cutoff=0.8)
    print(f"Number of removed collinear features: {len(redundant_features)}")
    print(f"Number of remaining mouse features: {len(free_mouse_feats) - len(redundant_features)}")
    dset = dset.drop(redundant_features, axis=1)

    # add the dataset to the dataset list
    free_mouse_datasets[thresh] = dset

# There is no need to get basic desc stats about all datasets (which was done after the mouse task dataset creation,
# because no cases were removed from the dataset)

#%%

# plot the kde plots for all datasets
for dset in free_mouse_datasets:
    print(f"KDE plots for: {dset}")
    # get the remaining task features
    remaining_features = [feat for feat in free_mouse_feats if feat in list(free_mouse_datasets[dset].columns)]
    multi_kde_plot(free_mouse_datasets[dset].loc[:, remaining_features + ["valence", "arousal"]], dset)


#%%

# plot a correlation heatmap for all datasets
for dset in free_mouse_datasets:
    print(f"Correlation Heatmap for: {dset}")
    remaining_features = [feat for feat in free_mouse_feats if feat in list(free_mouse_datasets[dset].columns)]
    correlation_heatmap(free_mouse_datasets[dset].loc[:, control_vars + remaining_features + ["valence", "arousal"]],
                        fig_size=(48, 38), font_scale=3.2, name=dset, add_text=True)

#%%

# inspect the relationship between the order and outcome and predictor variables more closely to get a descriptive
# feeling for a potential habituation effect throughout the study

# Do this using data visualization

# doing this for all variables in all datasets is a little bit much, therefore, do it for (randomly) selected variables
# in a randomly selected dataset (of course, it is also possible to select a target and dataset manually)
rand_free_dset_name, rand_free_dset = random.choice(list(free_mouse_datasets.items()))
random_target = random.choice([feat for feat in free_mouse_feats if feat in rand_free_dset.columns] +
                              ["valence", "arousal"])

#%%

# use the three different visualization approaches to descriptively check out the relationship between order and the
# target
plot_habituation_single_pars(rand_free_dset, random_target, rand_free_dset_name)

#%%
plot_habituation_multi_pars(rand_free_dset, random_target, rand_free_dset_name)

#%%
plot_habituation_all_pars(rand_free_dset, random_target, rand_free_dset_name)


#%%

# In the KDE-Plots it was noticable that not all mouse usage features as well as the dependent variables did now always
# follow a normal-distribution. One reviewer remarked that it might be better to transform the data before data analysis
# to make it more norma-like. We therefore applied rank-based inverse normal transformation to the data before feeding
# it into the mixed-level analysis. Here, use the transformation to check out how it changes the distribution (and
# possibly correlations) of the features

# loop the datasets to add the transformed features to each dataset
for dset in free_mouse_datasets:
    print(f"Transforming Features for Dataset: {dset}")
    # get the remaining features in the dataset
    remaining_features = [feat for feat in free_mouse_feats if feat in list(free_mouse_datasets[dset].columns)]
    # loop all remaining features and add the transformed features to the dataset
    for feat in remaining_features + ["valence", "arousal"]:
        free_mouse_datasets[dset][feat + "_trans"] = rank_inverse_normal_transform(free_mouse_datasets[dset][feat])

#%%

# create a kde plot for the transformed feature in each dataset
for dset in free_mouse_datasets:
    print(f"KDE plots after feature transformation: {dset}")
    trans_features = [feat + "_trans" for feat in free_mouse_feats if feat in list(free_mouse_datasets[dset].columns)]
    multi_kde_plot(free_mouse_datasets[dset].loc[:, trans_features + ["valence_trans", "arousal_trans"]], dset)

#%%

# create a correlation heatmap for the transformed features in each dataset
for dset in free_mouse_datasets:
    print(f"Heatmap after feature transformation: {dset}")
    trans_features = [feat + "_trans" for feat in free_mouse_feats if
                      feat in list(free_mouse_datasets[dset].columns)]
    correlation_heatmap(free_mouse_datasets[dset].loc[:, control_vars + trans_features + ["valence_trans", "arousal_trans"]],
                        fig_size=(48, 38), font_scale=3.2, name=dset, add_text=True)





