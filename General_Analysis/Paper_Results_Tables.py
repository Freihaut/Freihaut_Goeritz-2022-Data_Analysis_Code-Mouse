'''
Code to grab get some infos about the results and to create result table files for the research paper
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''

# package imports
import pandas as pd
import numpy as np
import pickle

#%%

# the variable and dataset names need to be changed for the result tables in the paper. The names in the code were
# conveniently chosen, but are not "pretty". This decision was made after the code was written, so we created a name
# change dictionary, which changes the names later
# the dictionary lists all "old names" and assigns them "new names", its a long dictionary and from a coding point of
# view not a very good solution
name_change_dict = {
    # control variables
    'timestamp': 'Timestamp', 'zoom': 'Zoom', 'screen_width': 'Screen Width',
    'screen_height': 'Screen Height', 'median_sampling_freq': 'Med. Samp. Frequency',
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
    'movement_distance': 'Movement Distance',
    "mo_ep_sd_abs_jerk_mean": 'Move Ep. (sd): Jerk (mean)',
    'mo_ep_mean_abs_jerk_sd': 'Move Ep. (mean): Jerk (sd)',
    'mo_ep_sd_abs_jerk_sd': 'Move Ep. (sd): Jerk (sd)',
    "mo_ep_sd_angle_mean": 'Move Ep. (sd): Angle (mean)',
    "mo_ep_mean_angle_sd": 'Move Ep. (mean): Angle (sd)',
    "mo_ep_mean_speed_sd": 'Move Ep. (mean): Speed (sd)',
    "mo_ep_sd_total_dist": 'Move Ep. (sd): Tot. Distance',
    "mo_ep_mean_y_flips": 'Move Ep. (mean): Y-Flips',
    'mo_ep_sd_y_flips': 'Move Ep. (sd): Y-Flips',
    'mo_ep_sd_x_flips': 'Move Ep. (sd): X-Flips',
    'mo_ep_sd_episode_duration': 'Move Ep. (sd): Eps. Duration',
    "lockscreen_episodes.": 'Num. of Lockscreen Eps.',
    # task datasets
    'cutoff_0': 'dur. cutoff',
    'iqr_out_2.5': 'IQR 2.5',
    'iqr_out_3.5': 'IQR 3.5',
    # for ml results
    'cutoff': 'dur. cutoff',
    'iqr_2.5': 'IQR 2.5',
    'iqr_3.5': 'IQR 3.5',
    # free mouse datasets
    '1000': '1s pause thresh',
    '2000': '2s pause thresh',
    '3000': '3s pause thresh',
    # for ml analysis
    'pause_thresh_1000': '1s pause thresh',
    'pause_thresh_2000': '2s pause thresh',
    'pause_thresh_3000': '3s pause thresh',
}


# %%

#######################
# Mixed Model Results #
#######################

# create a new dataframe with the following structure

#                                  Fixed Effect Model              |          Random Slope Model
#                   ---------------------------------------------- | ------------------------------------
#                   | -2DeltaLL | R²-cond | R²-marg | Coeff [CI]   | 2DeltaLL | R²-cond | R²-marg | Coeff [CI]
# --------------------------------------------------------------------------------------------------------------
#           | Dset1 |
# Predictor | Dset2 |
#           | Dset3 |


# Helper Functions that grab the data from the different result csv files and put them together into a dataset for
# the paper

# for the null models and baseline models
# inputs are the baseline task files as well as the random intercept only (null model) task files (only the diag files
# are needed, not the coefficient files)
def create_null_model_table(null_df):
    # results dict
    null_table = {}

    # get the AIC; R2_conditional, R2_marginal and ICC from the null model and the baseline model

    # loop every target and every dataset
    targets = null_df["dv"].unique()
    dsets = null_df["dframe"].unique()

    for target in targets:
        for dset in dsets:

            dset_name = name_change_dict[str(dset)]

            null_table[(target, dset_name)] = {}

            # get the target, dset subset from the dataframes
            null_diag = null_df.loc[(null_df["dv"] == target) & (null_df["dframe"] == dset)]

            # get the AIC, ICC, R2_conditional, R_2 marginal, R_2 cumulative for the null model
            null_table[(target, dset_name)][("Null Model", "AIC")] = null_diag["AIC"].values[0]
            null_table[(target, dset_name)][("Null Model", "ICC")] = null_diag["ICC"].values[0]
            null_table[(target, dset_name)][("Null Model", "R²-cond")] = null_diag["R2_conditional"].values[0]
            null_table[(target, dset_name)][("Null Model", "R²-marg")] = null_diag["R2_marginal"].values[0]
            null_table[(target, dset_name)][("Null Model", "R²-cum")] = null_diag["pseudo_R2"].values[0]

    return null_table


# helper to create a table for the model diagnostics of the predictor models
# inputs are the datasets of the fixed effect model diagnostics, the random slope model model diagnostics,
# and the model comparisons table
def create_pred_diag_table(fe_diag_df, rs_diag_df, comparison_df):
    # results table dict
    pred_diag_table = {}

    # loop every predictor for every data set and every dependent variable, isolate the desired results and save them
    sp_targets = fe_diag_df["dv"].unique()
    sp_dsets = fe_diag_df["dframe"].unique()
    sp_preds = fe_diag_df["iv"].unique()

    # Loop all combinations to fill the single pred_table in order to create the results dataframe
    for target in sp_targets:
        pred_diag_table[target] = {}
        for pred in sp_preds:
            for dset in sp_dsets:
                # extract the relevant info from the result tables and save them in the dictionary

                # first check if the target, pred, dset combination exists
                # to do so, extract the combination from the random intercept model and get the length (should be
                # greater than 0)
                ri_diag = fe_diag_df.loc[(fe_diag_df["dv"] == target) &
                                         (fe_diag_df["iv"] == pred) &
                                         (fe_diag_df["dframe"] == dset)]

                if len(ri_diag) > 0:
                    pred_name = name_change_dict[pred]
                    dset_name = name_change_dict[str(dset)]
                    pred_diag_table[target][(pred_name, dset_name)] = {}

                    # get the relevant model comparison data to grab the results of the log likelihood ratio test
                    likelihoodratio_data = comparison_df.loc[(comparison_df["dv"] == target) &
                                                             (comparison_df["iv"] == pred) &
                                                             (comparison_df["dframe"] == dset)].reset_index()

                    # Fixed Effect Model Diagnostics (-2ΔLL, AIC, R²-cond, R²-marg, R²-cum)
                    pred_diag_table[target][(pred_name, dset_name)][('Fixed Effect Model', u'-2ΔLL')] = \
                        likelihoodratio_data.at[1, 'p']
                    pred_diag_table[target][(pred_name, dset_name)][('Fixed Effect Model', 'AIC')] = \
                        ri_diag['AIC'].values[0]
                    pred_diag_table[target][(pred_name, dset_name)][('Fixed Effect Model', 'R²-cond')] = \
                        ri_diag['R2_conditional'].values[0]
                    pred_diag_table[target][(pred_name, dset_name)][('Fixed Effect Model', 'R²-marg')] = \
                        ri_diag['R2_marginal'].values[0]
                    pred_diag_table[target][(pred_name, dset_name)][('Fixed Effect Model', 'R²-cum')] = \
                        ri_diag['pseudo_R2'].values[0]

                    # Random Slope Model Diagnostics (-2ΔLL, AIC, R²-cond, R²-marg, R²-cum)
                    rs_diag = rs_diag_df.loc[(rs_diag_df["dv"] == target) &
                                             (rs_diag_df["iv"] == pred) &
                                             (rs_diag_df["dframe"] == dset)]

                    pred_diag_table[target][(pred_name, dset_name)][('Rand. Slope Model', u'-2ΔLL')] = \
                        likelihoodratio_data.at[2, 'p']
                    pred_diag_table[target][(pred_name, dset_name)][('Rand. Slope Model', 'AIC')] = \
                        rs_diag['AIC'].values[0]
                    pred_diag_table[target][(pred_name, dset_name)][('Rand. Slope Model', 'R²-cond')] = \
                        rs_diag['R2_conditional'].values[0]
                    pred_diag_table[target][(pred_name, dset_name)][('Rand. Slope Model', 'R²-marg')] = \
                        rs_diag['R2_marginal'].values[0]
                    pred_diag_table[target][(pred_name, dset_name)][('Rand. Slope Model', 'R²-cum')] = \
                        rs_diag['pseudo_R2'].values[0]

    return pred_diag_table


# helper to create a table for the effect coefficients of the predictor models
# inputs are the datasets of the fixed effect model coefficients, the standardized fixed effect model coefficients
# the random slope model coefficients, and the standardized random slope model coefficients
def create_pred_coefficients_table(fe_coefficient_df, fe_std_coefficient_df, rs_coefficient_df, rs_std_coefficient_df):

    # results table dict
    pred_coeff_table = {}

    # loop every predictor for every data set and every dependent variable, isolate the desired results and save them
    targets = fe_coefficient_df["dv"].unique()
    dsets = fe_coefficient_df["dframe"].unique()
    preds = fe_coefficient_df["iv"].unique()

    # Loop all combinations to fill the single pred_table in order to create the results dataframe
    for target in targets:
        pred_coeff_table[target] = {}
        for pred in preds:
            for dset in dsets:

                # extract the relevant info from the result tables and save them in the dictionary

                # first check if the target, pred, dset combination exists
                # to do so, extract the combination from the random intercept model and get the length (should be
                # greater than 0)
                fe_coeffs = fe_coefficient_df.loc[(fe_coefficient_df["dv"] == target) &
                                                  (fe_coefficient_df["iv"] == pred) &
                                                  (fe_coefficient_df["dframe"] == dset)]

                if len(fe_coeffs) > 0:
                    pred_name = name_change_dict[pred]
                    dset_name = name_change_dict[str(dset)]

                    # get the standardized fixed effect coefficients for the predictor + dataset combination
                    fe_std_coeffs = fe_std_coefficient_df.loc[(fe_std_coefficient_df["dv"] == target) &
                                                                     (fe_std_coefficient_df["iv"] == pred) &
                                                                     (fe_std_coefficient_df["dframe"] == dset)]

                    # get the random slope coefficients for the predictor + dataset combination
                    rs_coeffs = rs_coefficient_df.loc[(rs_coefficient_df["dv"] == target) &
                                                      (rs_coefficient_df["iv"] == pred) &
                                                      (rs_coefficient_df["dframe"] == dset)]

                    # get the standardized random slope coefficients for the predictor + dataset combination
                    rs_std_coeffs = rs_std_coefficient_df.loc[(rs_std_coefficient_df["dv"] == target) &
                                                              (rs_std_coefficient_df["iv"] == pred) &
                                                              (rs_std_coefficient_df["dframe"] == dset)]

                    # -- First, get all within-predictor information -- #

                    # first, extract the within predictor coefficients
                    pred_coeff_table[target][(pred_name, dset_name, "within-Effect")] = {}

                    # Fixed Effect Model

                    # get the estimator row
                    fe_within_coeffs = fe_coeffs.loc[fe_coeffs["term"] == pred + "_within"]

                    # extract the Coefficient and Confidence Interval Estimate
                    pred_coeff_table[target][(pred_name, dset_name, "within-Effect")][
                        ('Fixed Effect Model', 'Fixed Effect Est.')] = str(
                        np.round(fe_within_coeffs["estimate"].values[0], 2)) + "\n[" \
                                                                          + str(
                        np.round(fe_within_coeffs["conf.low"].values[0], 2)) + ", " + str(
                        np.round(fe_within_coeffs["conf.high"].values[0], 2)) + "]"

                    # get the p-value
                    pred_coeff_table[target][(pred_name, dset_name, "within-Effect")][
                        ('Fixed Effect Model', 'p-value')] = fe_within_coeffs["p.value"].values[0]

                    # Standardized Fixed Effect Model
                    fe_within_std_coeffs = fe_std_coeffs.loc[(fe_std_coefficient_df["Component"] == "within")]

                    # extract the coefficient and the confidence interval
                    pred_coeff_table[target][(pred_name, dset_name, "within-Effect")][
                        ('Fixed Effect Model', 'Std. Fixed Effect Est.')] = str(
                        np.round(fe_within_std_coeffs["Std_Coefficient"].values[0], 2)) + "\n[" \
                                                                       + str(
                        np.round(fe_within_std_coeffs["CI_low"].values[0], 2)) + ", " + str(
                        np.round(fe_within_std_coeffs["CI_high"].values[0], 2)) + "]"

                    # Random Slope Model
                    rs_within_coeffs = rs_coeffs.loc[rs_coeffs["term"] == pred + "_within"]

                    pred_coeff_table[target][(pred_name, dset_name, "within-Effect")][
                        ('Rand. Slope Model', 'Fixed Effect Est.')] = str(
                        np.round(rs_within_coeffs["estimate"].values[0], 2)) + "\n[" \
                                                                       + str(
                        np.round(rs_within_coeffs["conf.low"].values[0], 2)) + ", " + str(
                        np.round(rs_within_coeffs["conf.high"].values[0], 2)) + "]"

                    # get the p-value
                    pred_coeff_table[target][(pred_name, dset_name, "within-Effect")][
                        ('Rand. Slope Model', 'p-value')] = rs_within_coeffs["p.value"].values[0]

                    # Standardized Random Slope Model
                    rs_within_std_coeffs = rs_std_coeffs.loc[(rs_std_coeffs["Component"] == "within")]

                    pred_coeff_table[target][(pred_name, dset_name, "within-Effect")][
                        ('Rand. Slope Model', 'Std. Fixed Effect Est.')] = str(
                        np.round(rs_within_std_coeffs["Std_Coefficient"].values[0], 2)) + "\n[" \
                                                                            + str(
                        np.round(rs_within_std_coeffs["CI_low"].values[0], 2)) + ", " + str(
                        np.round(rs_within_std_coeffs["CI_high"].values[0], 2)) + "]"

                    # Random Slope Random Estimate
                    rs_rand_coeff = rs_coeffs.loc[rs_coeffs["term"] == 'sd__' + pred + "_within"]
                    pred_coeff_table[target][(pred_name, dset_name, "within-Effect")][
                        ('Rand. Slope Model', 'Rand. Effect Est.')] = rs_rand_coeff["estimate"].values[0]

                    # -- Second, get the between-effect coefficients -- #

                    # first, extract the within predictor coefficients
                    pred_coeff_table[target][(pred_name, dset_name, "between-Effect")] = {}

                    # Fixed Effect Model

                    # get the estimator row
                    fe_between_coeffs = fe_coeffs.loc[fe_coeffs["term"] == pred + "_between"]

                    # extract the Coefficient and Confidence Interval Estimate
                    pred_coeff_table[target][(pred_name, dset_name, "between-Effect")][
                        ('Fixed Effect Model', 'Fixed Effect Est.')] = str(
                        np.round(fe_between_coeffs["estimate"].values[0], 2)) + "\n[" \
                                                                       + str(
                        np.round(fe_between_coeffs["conf.low"].values[0], 2)) + ", " + str(
                        np.round(fe_between_coeffs["conf.high"].values[0], 2)) + "]"

                    # get the p-value
                    pred_coeff_table[target][(pred_name, dset_name, "between-Effect")][
                        ('Fixed Effect Model', 'p-value')] = fe_between_coeffs["p.value"].values[0]

                    # Standardized Fixed Effect Model
                    fe_between_std_coeffs = fe_std_coeffs.loc[(fe_std_coefficient_df["Component"] == "between")]

                    # extract the coefficient and the confidence interval
                    pred_coeff_table[target][(pred_name, dset_name, "between-Effect")][
                        ('Fixed Effect Model', 'Std. Fixed Effect Est.')] = str(
                        np.round(fe_between_std_coeffs["Std_Coefficient"].values[0], 2)) + "\n[" \
                                                                            + str(
                        np.round(fe_between_std_coeffs["CI_low"].values[0], 2)) + ", " + str(
                        np.round(fe_between_std_coeffs["CI_high"].values[0], 2)) + "]"

                    # Random Slope Model
                    rs_between_coeffs = rs_coeffs.loc[rs_coeffs["term"] == pred + "_between"]

                    pred_coeff_table[target][(pred_name, dset_name, "between-Effect")][
                        ('Rand. Slope Model', 'Fixed Effect Est.')] = str(
                        np.round(rs_between_coeffs["estimate"].values[0], 2)) + "\n[" \
                                                                      + str(
                        np.round(rs_between_coeffs["conf.low"].values[0], 2)) + ", " + str(
                        np.round(rs_between_coeffs["conf.high"].values[0], 2)) + "]"

                    # get the p-value
                    pred_coeff_table[target][(pred_name, dset_name, "between-Effect")][
                        ('Rand. Slope Model', 'p-value')] = rs_between_coeffs["p.value"].values[0]

                    # Standardized Random Slope Model
                    rs_between_std_coeffs = rs_std_coeffs.loc[(rs_std_coeffs["Component"] == "between")]

                    pred_coeff_table[target][(pred_name, dset_name, "between-Effect")][
                        ('Rand. Slope Model', 'Std. Fixed Effect Est.')] = str(
                        np.round(rs_between_std_coeffs["Std_Coefficient"].values[0], 2)) + "\n[" \
                                                                           + str(
                        np.round(rs_between_std_coeffs["CI_low"].values[0], 2)) + ", " + str(
                        np.round(rs_between_std_coeffs["CI_high"].values[0], 2)) + "]"

                    # the between effect has no random term because it is a level 2 predictor

    return pred_coeff_table


# helper function to create an ICC table for every mouse usage feature (as part of the descriptive statistics)
def create_pred_icc_table(icc_results_df):

    # results dict
    icc_table = {}

    # get the unique targets and datasets in the icc df
    targets = icc_results_df["iv"].unique()
    dsets = icc_results_df["dframe"].unique()

    for target in targets:
        for dset in dsets:

            # extract the relevant info from the dataset
            icc_row = icc_results_df.loc[(icc_results_df["iv"] == target) & (icc_results_df["dframe"] == dset)]

            if len(icc_row) > 0:

                icc_table[(name_change_dict[target], name_change_dict[str(dset)])] = {}

                icc_table[(name_change_dict[target], name_change_dict[str(dset)])]["ICC"] = icc_row["ICC"].iloc[0]

    return icc_table


# %%

# -----------
# Task Data -
# -----------

# Null-Model Results and Baseline Results
# ---------------------------------------

# import the datasets
task_null_diag = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis/Results_NEW/Task_results_ri_diag.csv")

#%%

# get the results table
task_null_diag_table = create_null_model_table(task_null_diag)

#%%

# Predictor Model results
# ------------------------------

# import all results from the csv file (there are 5 result files)
task_pred_fe_coeffs = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis/Results_NEW/Task_results_fe_coeffs.csv")
task_pred_fe_diag = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis/Results_NEW/Task_results_fe_diag.csv")
task_pred_fe_std_coeffs = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis/Results_NEW/Task_results_fe_std_coeffs.csv")

task_pred_rs_coeffs = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis/Results_NEW/Task_results_rs_coeffs.csv")
task_pred_rs_diag = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis/Results_NEW/Task_results_rs_diag.csv")
task_pred_rs_std_coeffs = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis/Results_NEW/Task_results_rs_std_coeffs.csv")

task_pred_model_comp = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis/Results_NEW/Task_results_model_comparison.csv")

#%%

# Get the model diagnostics result table for the single predictor models
task_pred_diag_table = create_pred_diag_table(task_pred_fe_diag, task_pred_rs_diag, task_pred_model_comp)

#%%

# get the effect coefficient result table for the single predictor models
task_pred_coefficient_table = create_pred_coefficients_table(task_pred_fe_coeffs, task_pred_fe_std_coeffs,
                                                             task_pred_rs_coeffs, task_pred_rs_std_coeffs)

#%%

# get the ICC table for all mouse usage features (part of the descriptive stats)

# import the ICC dataset
task_icc_df = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis/Results_NEW/Task_results_MousePred_ICC.csv")
task_icc_table = create_pred_icc_table(task_icc_df)


# %%

# -----------------------
# Free Mouse Usage Data -
# -----------------------

# Null-Model Results and Baseline Results
# ---------------------------------------

# import the datasets
free_null_mod_diag = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Free-Mouse_Analysis/Results_NEW/FreeMouse_results_ri_diag.csv")

#%%

# get the results table
free_null_result_table = create_null_model_table(free_null_mod_diag)

#%%

# Single Predictor Model results
# ------------------------------

# import all results from the csv file (there are 5 result files)
# import all results from the csv file (there are 5 result files)
free_pred_fe_coeffs = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Free-Mouse_Analysis/Results_NEW/FreeMouse_results_fe_coeffs.csv")
free_pred_fe_diag = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Free-Mouse_Analysis/Results_NEW/FreeMouse_results_fe_diag.csv")
free_pred_fe_std_coeffs = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Free-Mouse_Analysis/Results_NEW/FreeMouse_results_fe_std_coeffs.csv")

free_pred_rs_coeffs = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Free-Mouse_Analysis/Results_NEW/FreeMouse_results_rs_coeffs.csv")
free_pred_rs_diag = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Free-Mouse_Analysis/Results_NEW/FreeMouse_results_rs_diag.csv")
free_pred_rs_std_coeffs = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Free-Mouse_Analysis/Results_NEW/FreeMouse_results_rs_std_coeffs.csv")

free_pred_model_comp = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Free-Mouse_Analysis/Results_NEW/FreeMouse_results_model_comparison.csv")

#%%

# Get the model diagnostics result table for the single predictor models
free_pred_diag_table = create_pred_diag_table(free_pred_fe_diag, free_pred_rs_diag, free_pred_model_comp)

#%%

# get the effect coefficient result table for the single predictor models
free_pred_coefficient_table = create_pred_coefficients_table(free_pred_fe_coeffs, free_pred_fe_std_coeffs,
                                                             free_pred_rs_coeffs, free_pred_rs_std_coeffs)

#%%

# get the ICC table for all mouse usage features (part of the descriptive stats)

# import the ICC dataset
free_icc_df = pd.read_csv("C:/Users/Paul/Desktop/Data_Analysis/Free-Mouse_Analysis/Results_NEW/FreeMouse_results_MousePred_ICC.csv")
free_icc_table = create_pred_icc_table(free_icc_df)


#%%

# Get some descriptive stats about the models, because manually reading them from the tables is tideous
# -----------------------------------------------------------------------------------------------------

# Get info about

# - Coefficient estimates, their p-values and confidence intervals

# simple helper function to print descriptive stats about the ICC across the datasets for a specified target variable
def describe_null_model(results_table, target):

    # create the relevant dataframe from the null model results table
    null_mod = pd.DataFrame.from_dict(results_table, orient='index').rename_axis(["target", "dset"]).round(2).loc[target, :].droplevel(0, axis=1)

    # simply describe the ICC column of the dataframe for the specified target variable to get the relevant descriptive
    # stats
    print(f"ICC Descriptives for target {target}")
    print(null_mod["ICC"].describe())

    return


# helper function to print infos about the model comparisons
def describe_model_comparison(null_diag_table, pred_diag_table, target):

    # create the relevant dataframe from the null_diag_table
    null_diag = pd.DataFrame.from_dict(null_diag_table, orient='index').rename_axis(["target", "dset"]).round(4).loc[
               target, :]

    # create the relevant dataframe from the pred_diag_table
    pred_diag = pd.DataFrame.from_dict(pred_diag_table[target], orient='index').rename_axis(["target", "dset"])
    # replace R²-marg values with NaN in the random slope model if no R²-cond could be computed for the row due to
    # missing variance in the slope (use a simple hack and add the nan column first and then subtract it again)
    pred_diag[("Rand. Slope Model", "R²-marg")] = pred_diag[("Rand. Slope Model", "R²-marg")] + pred_diag[("Rand. Slope Model", "R²-cond")]
    pred_diag[("Rand. Slope Model", "R²-marg")] = pred_diag[("Rand. Slope Model", "R²-marg")] - pred_diag[("Rand. Slope Model", "R²-cond")]

    # merge both datasets
    merged_df = pred_diag.join(null_diag, how="inner")

    # count the number of calculated models
    print(f"Number of calculated models: {len(merged_df)}\n")

    # count the number of significant loglikelihood ratio tests between the null model and the fixed effect model
    print("Num. of significant Log Likelihood Tests between the fixed effect model and the null model")
    print(len(merged_df.loc[merged_df[("Fixed Effect Model", "-2ΔLL")] < 0.05]))
    print("After Bonferroni correction")
    print(len(merged_df.loc[merged_df[("Fixed Effect Model", "-2ΔLL")] < (0.05 / len(merged_df))]))
    print("\n")

    # count the number of significant loglikelhood ratio tests between the random slope model and the fixed effect model
    print("Num. of significant Log Likelihood Tests between the random slope model and the fixed effect model")
    print(len(merged_df.loc[merged_df[("Rand. Slope Model", "-2ΔLL")] < 0.05]))
    print("After Bonferroni correction")
    print(len(merged_df.loc[merged_df[("Rand. Slope Model", "-2ΔLL")] < (0.05 / len(merged_df))]))
    print("\n")

    # calculate the R²-value differences between the models
    merged_df[("Comp", "F-N_R2_cond")] = merged_df[("Fixed Effect Model", "R²-cond")] - merged_df[("Null Model", "R²-cond")]
    merged_df[("Comp", "F-N_R2_marg")] = merged_df[("Fixed Effect Model", "R²-marg")] - merged_df[("Null Model", "R²-marg")]
    merged_df[("Comp", "F-N_R2_cum")] = merged_df[("Fixed Effect Model", "R²-cum")] - merged_df[("Null Model", "R²-cum")]

    merged_df[("Comp", "S-F_R2_cond")] = merged_df[("Rand. Slope Model", "R²-cond")] - merged_df[("Fixed Effect Model", "R²-cond")]
    merged_df[("Comp", "S-F_R2_marg")] = merged_df[("Rand. Slope Model", "R²-marg")] - merged_df[("Fixed Effect Model", "R²-marg")]
    merged_df[("Comp", "S-F_R2_cum")] = merged_df[("Rand. Slope Model", "R²-cum")] - merged_df[("Fixed Effect Model", "R²-cum")]

    # get descriptives about the R²-changes
    print(merged_df[("Comp", "F-N_R2_cond")].describe())
    print("\n")
    print(merged_df[("Comp", "F-N_R2_marg")].describe())
    print("\n")
    print(merged_df[("Comp", "F-N_R2_cum")].describe())
    print("\n")

    # we need to drop some columns which had propblems with the R²-cond and R²-marg calculation due to missing variance
    # in the random slope
    print(merged_df[("Comp", "S-F_R2_cond")].dropna(how="all").describe())
    print("\n")
    print(merged_df[("Comp", "S-F_R2_marg")].dropna(how="all").describe())
    print("\n")
    print(merged_df[("Comp", "S-F_R2_cum")].describe())
    print("\n")

    return


# simple helper function to describe the model coefficients
def describe_coefficients(coeff_table, target):

    # create a dataframe for the target variable
    coeff_df = pd.DataFrame.from_dict(coeff_table[target], orient='index').rename_axis(["target", "dset", "effect"])

    # get the between effect coefficients
    between_coeffs = coeff_df.xs("between-Effect", axis=0, level=2, drop_level=False)
    # count the number of significant between effects
    print(f"Number of significant between effects in the fixed effect model: "
          f"{len(between_coeffs.loc[between_coeffs[('Fixed Effect Model', 'p-value')] < 0.05])}")
    print(f"After Bonferroni Correction: "
          f"{len(between_coeffs.loc[between_coeffs[('Fixed Effect Model', 'p-value')] < (0.05/len(between_coeffs))])}")

    print(f"Number of significant between effects in the random slope model: "
          f"{len(between_coeffs.loc[between_coeffs[('Rand. Slope Model', 'p-value')] < 0.05])}")
    print(f"After Bonferroni Correction: "
          f"{len(between_coeffs.loc[between_coeffs[('Rand. Slope Model', 'p-value')] < (0.05 / len(between_coeffs))])}")

    # do the same with the within-effects
    within_coeffs = coeff_df.xs("within-Effect", axis=0, level=2, drop_level=False)
    # count the number of significant between effects
    print(f"Number of significant within effects in the fixed effect model: "
          f"{len(within_coeffs.loc[within_coeffs[('Fixed Effect Model', 'p-value')] < 0.05])}")
    print(f"After Bonferroni Correction: "
          f"{len(within_coeffs.loc[within_coeffs[('Fixed Effect Model', 'p-value')] < (0.05 / len(within_coeffs))])}")

    print(f"Number of significant within effects in the random slope model: "
          f"{len(within_coeffs.loc[within_coeffs[('Rand. Slope Model', 'p-value')] < 0.05])}")
    print(f"After Bonferroni Correction: "
          f"{len(within_coeffs.loc[within_coeffs[('Rand. Slope Model', 'p-value')] < (0.05 / len(within_coeffs))])}")

    # finally, get the range of the random slope of the within-effect
    print("Descriptive Stats about the random slope effect")
    print(within_coeffs[('Rand. Slope Model', 'Rand. Effect Est.')].describe())

    return

#%%

# -- Now describe the results for all result datasets -- #

# ..Mouse Task Analysis.. #
print("Printing summarized results for the mixed model analysis of the MOUSE TASK data")

#%%

# Get the null model results

describe_null_model(task_null_diag_table, "arousal")

#%%

describe_null_model(task_null_diag_table, "valence")

#%%

# Get the Model Comparison Results
describe_model_comparison(task_null_diag_table, task_pred_diag_table, "arousal")

#%%

describe_model_comparison(task_null_diag_table, task_pred_diag_table, "valence")

#%%

# Get the Model Coefficient Results
describe_coefficients(task_pred_coefficient_table, "arousal")

#%%

describe_coefficients(task_pred_coefficient_table, "valence")

#%%

# ..Free Mouse Analysis.. #
print("Printing summarized results for the mixed model analysis of the FREE MOUSE data")

#%%

# Get the null model results

describe_null_model(free_null_result_table, "arousal")

#%%

describe_null_model(free_null_result_table, "valence")

#%%

# Get the Model Comparison Results
describe_model_comparison(free_null_result_table, free_pred_diag_table, "arousal")

#%%

describe_model_comparison(free_null_result_table, free_pred_diag_table, "valence")

#%%

# Get the Model Coefficient Results
describe_coefficients(free_pred_coefficient_table, "arousal")

#%%

describe_coefficients(free_pred_coefficient_table, "valence")

#%%

############################
# Machine Learning Results #
############################

# create a new dataframe with the following structure

#                                 Baseline           Full Mod
#                            ------------------  ---------------
#           Train/Test Shape |  R² | MSE | MAE   R² | MSE | MAE
# ----------------------------------------------------------------
#  Dset1 |
#  Dset2 |
#  Dset3 |


# helper function to extract the relevant results from the results data set
def ml_results_table(result_files):

    result_table = {}

    # get the target variables to create separate tables per target
    targets = ["arousal", "valence"]

    # loop the targets
    for target in targets:
        result_table[target] = {}
        # loop the results
        for result in result_files:
            # check the target variable to see if we need the regression or classification results
            if target in result:
                # get the dset name
                dset_name = name_change_dict[result.split('_', 1)[1]]
                result_table[target][dset_name] = {}
                # # add the dataset shapes
                result_table[target][dset_name][("Num. Samples", "Train Data")] = result_files[result]['dset_shapes']['train_shape'][0]
                result_table[target][dset_name][("Num. Samples", "Test Data")] = result_files[result]['dset_shapes']['test_shape'][0]

                # add the baseline results
                result_table[target][dset_name][("Null Model", "num preds.")] = len(result_files[result]['predictors']['baseline'])
                result_table[target][dset_name][("Null Model", "R²-score")] = result_files[result]['baseline_results']['scores']['r2']
                result_table[target][dset_name][("Null Model", "MSE")] = result_files[result]['baseline_results']['scores']['mse']
                result_table[target][dset_name][("Null Model", "MAE")] = result_files[result]['baseline_results']['scores']['mae']
                # add the full model results
                result_table[target][dset_name][("Full Model", 'num. preds.')] = len(result_files[result]['predictors']['mouse_only'])
                result_table[target][dset_name][("Full Model", "R²-score")] = result_files[result]['full_model_results']['scores']['r2']
                result_table[target][dset_name][("Full Model", "MSE")] = result_files[result]['full_model_results']['scores']['mse']
                result_table[target][dset_name][("Full Model", "MAE")] = result_files[result]['full_model_results']['scores']['mae']

    return result_table


# %%

# Mouse Usage Task
# ----------------

with open('./Mouse-Task_Analysis/Mouse_task_ML_Results.p', 'rb') as handle:
    task_ml_results = pickle.load(handle)

# %%

task_ml_result_table = ml_results_table(task_ml_results)

# %%

# Free Mouse Usage
# -----------------

with open("./Free-Mouse_Analysis/Free_Mouse_ML_results.p", "rb") as handle:
    free_ml_results = pickle.load(handle)

# %%

free_ml_results_table = ml_results_table(free_ml_results)


# %%

# helper function to get some descriptive results of the machine learning analysis
def get_ml_descriptives(ml_results, target):
    # create the relevant dataframe from the null model results table
    ml_df = pd.DataFrame.from_dict(ml_results[target], orient='index').rename_axis(["target"])

    # now get simple descriptive stats about selected columns
    print(f"Average Null Model R²: {ml_df[('Null Model', 'R²-score')].mean()}")
    print(f"Average Null Model MSE: {ml_df[('Null Model', 'MSE')].mean()}")
    print(f"Average Null Model MAE: {ml_df[('Null Model', 'MAE')].mean()}")
    print("\n")
    print(f"Average Full Model R²: {ml_df[('Full Model', 'R²-score')].mean()}")
    print(f"Average Full Model MSE: {ml_df[('Full Model', 'MSE')].mean()}")
    print(f"Average Full Model MAE: {ml_df[('Full Model', 'MAE')].mean()}")

    return

#%%

# Get desc. stats for the mouse task ml analysis results

get_ml_descriptives(task_ml_result_table, "arousal")

#%%

get_ml_descriptives(task_ml_result_table, "valence")

#%%

# Get desc. stats for the free mouse ml analysis results

get_ml_descriptives(free_ml_results_table, "arousal")

# %%

get_ml_descriptives(free_ml_results_table, "valence")

#%%

############################
# Create Tables per Target #
############################

# create the result tables and save them as csv files (they will need further manual processing in order to fit the
# APA format)


# helper function to save the output table as csv files, the baseline table and single predictor/interaction table
# need separate helper functions
def save_null_model(table_dict, filename):

    # convert the null model & baseline model dictionary to a dataframe
    df = pd.DataFrame.from_dict(table_dict, orient='index').rename_axis(["target", "dset"]).round(4)

    # save it as an excel file
    df.to_excel(filename + '.xlsx')


def save_model_coeffs(table_dict, filename):
    # create an excel file with separate sheets that contain the results for each target variable
    with pd.ExcelWriter(filename + ".xlsx") as writer:
        # the table dict contains a separate dictionary for each target variable (alternative would be to create one
        # large multiindex dataframe)
        for target in table_dict:
            # convert the target table dict into a dataframe
            table_df = pd.DataFrame.from_dict(table_dict[target], orient='index').rename_axis(["targt", "dset", "pred"]).round(4)

            # change p-values to significance strings
            conditions0 = [
                table_df[("Fixed Effect Model", "p-value")] < (0.05 / (len(table_df[("Fixed Effect Model", "p-value")])/2)),
                table_df[("Fixed Effect Model", "p-value")] < 0.05,
                table_df[("Fixed Effect Model", "p-value")] >= 0.05
            ]

            conditions1 = [
                table_df[("Rand. Slope Model", "p-value")] < (0.05 / (len(table_df[("Rand. Slope Model", "p-value")])/2)),
                table_df[("Rand. Slope Model", "p-value")] < 0.05,
                table_df[("Rand. Slope Model", "p-value")] >= 0.05
            ]

            choices = ["< .05*", "< .05", "n.s."]

            # change p-values of both columns
            table_df[("Fixed Effect Model", "p-value")] = np.select(conditions0, choices)
            table_df[("Rand. Slope Model", "p-value")] = np.select(conditions1, choices)

            # save the table df as a csv file
            table_df.to_excel(writer, sheet_name=target)


def save_model_diag(table_dict, filename):
    # create an excel file with separate sheets that contain the results for each target variable
    with pd.ExcelWriter(filename + ".xlsx") as writer:
        # the table dict contains a separate dictionary for each target variable (alternative would be to create one
        # large multiindex dataframe)
        for target in table_dict:
            # convert the target table dict into a dataframe
            table_df = pd.DataFrame.from_dict(table_dict[target], orient='index').rename_axis(["pred", "dset"]).round(4)
            # hacky way to remove R²-marg values if no R²-cond values were calculated
            table_df[("Rand. Slope Model", "R²-marg")] = table_df[("Rand. Slope Model", "R²-marg")] + table_df[
                ("Rand. Slope Model", "R²-cond")]
            table_df[("Rand. Slope Model", "R²-marg")] = table_df[("Rand. Slope Model", "R²-marg")] - table_df[
                ("Rand. Slope Model", "R²-cond")]

            # change p-values to significance strings
            conditions0 = [
                table_df[("Fixed Effect Model", "-2ΔLL")] < (0.05 / len(table_df[("Fixed Effect Model", "-2ΔLL")])),
                table_df[("Fixed Effect Model", "-2ΔLL")] < 0.05,
                table_df[("Fixed Effect Model", "-2ΔLL")] >= 0.05
            ]

            conditions1 = [
                table_df[("Rand. Slope Model", "-2ΔLL")] < (0.05 / len(table_df[("Rand. Slope Model", "-2ΔLL")])),
                table_df[("Rand. Slope Model", "-2ΔLL")] < 0.05,
                table_df[("Rand. Slope Model", "-2ΔLL")] >= 0.05
            ]

            choices = ["< .05*", "< .05", "n.s."]

            # change p-values of both columns
            table_df[("Fixed Effect Model", "-2ΔLL")] = np.select(conditions0, choices)
            table_df[("Rand. Slope Model", "-2ΔLL")] = np.select(conditions1, choices)

            # save the table df as a csv file
            table_df.to_excel(writer, sheet_name=target)


def save_ml_output(table_dict, filename):
    # create an excel file with separate sheets that contain the results for each target variable
    with pd.ExcelWriter(filename + ".xlsx") as writer:
        # the table dict contains a separate dictionary for each target variable
        for target in table_dict:
            # convert the target table dict into a dataframe
            table_df = pd.DataFrame.from_dict(table_dict[target], orient='index').rename_axis(["dset"]).round(2)
            # save the table df as a csv file
            table_df.to_excel(writer, sheet_name=target)


#%%

# save the results that should be highlighted in the paper as csv files
save_null_model(task_null_diag_table, "Task_Null_Models_resTable")
save_null_model(task_icc_table, "Task_ICC_Models_resTable")

save_null_model(free_null_result_table, "Free_Null_Models_resTable")
save_null_model(free_icc_table, "Free_ICC_Models_resTable")

#%%

save_model_diag(task_pred_diag_table, "Task_Pred_Diag_resTable")
save_model_diag(free_pred_diag_table, "Free_Pred_Diag_resTable")

#%%

save_model_coeffs(task_pred_coefficient_table, "Task_Pred_Coeffs_resTable")
save_model_coeffs(free_pred_coefficient_table, "Free_Pred_Coeffs_resTable")

#%%

save_ml_output(task_ml_result_table, "Task_ML_resTable")
save_ml_output(free_ml_results_table, "Free_ML_resTable")



