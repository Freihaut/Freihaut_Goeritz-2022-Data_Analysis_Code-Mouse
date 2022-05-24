'''
Code to grab get some infos about the results and to create result table files for the research paper
The code was run in Pycharm with scientific mode turned on. The #%% symbol separates the code into cells, which
can be run separately from another (similar to a jupyter notebook)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'''

# package imports
import pandas as pd
import numpy as np
import json

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
    "mo_ep_sd_abs_jerk_mean": 'Move Ep. (sd): Jerk (mean)',
    "mo_ep_sd_angle_mean": 'Move Ep. (sd): Angle (mean)',
    "mo_ep_mean_angle_sd": 'Move Ep. (mean): Angle (sd)',
    "mo_ep_mean_speed_sd": 'Move Ep. (mean): Speed (sd)',
    "mo_ep_sd_total_dist": 'Move Ep. (sd): Tot. Distance',
    "mo_ep_mean_y_flips": 'Move Ep. (mean): Y-Flips',
    "lockscreen_episodes.": 'Num. of Lockscreen Eps.',
    # task datasets
    'by_sample_cutoffNA': 'dur. cutoff & std. by sample',
    'by_sample_iqr_out2.5': 'IQR 2.5 & std. by sample',
    'by_sample_iqr_out3.5': 'IQR 3.5 & std. by sample',
    'by_participant_cutoffNA': 'dur. cutoff & std. by par',
    'by_participant_iqr_out2.5': 'IQR 2.5 & std. by par',
    'by_participant_iqr_out3.5': 'IQR 3.5 & std. by par',
    # for ml results
    'by_sample_cutoff': 'dur. cutoff & std. by sample',
    'by_sample_iqr_2.5': 'IQR 2.5 & std. by sample',
    'by_sample_iqr_3.5': 'IQR 3.5 & std. by sample',
    'by_participant_cutoff': 'dur. cutoff & std. by par',
    'by_participant_iqr_2.5': 'IQR 2.5 & std. by par',
    'by_participant_iqr_3.5': 'IQR 3.5 & std. by par',
    # free mouse datasets
    'p_thresh_1000_by_sample': '1s pause & std. by sample',
    'p_thresh_2000_by_sample': '2s pause & std. by sample',
    'p_thresh_3000_by_sample': '3s pause & std. by sample',
    'p_thresh_1000_by_participant': '1s pause & std. by par',
    'p_thresh_2000_by_participant': '2s pause & std. by par',
    'p_thresh_3000_by_participant': '3s pause & std. by par',
    # for ml results
    '1000_by_sample': '1s pause & std. by sample',
    '2000_by_sample': '2s pause & std. by sample',
    '3000_by_sample': '3s pause & std. by sample',
    '1000_by_participant': '1s pause & std. by par',
    '2000_by_participant': '2s pause & std. by par',
    '3000_by_participant': '3s pause & std. by par',
}


# %%

#######################
# Mixed Model Results #
#######################

# create a new dataframe with the following structure

#                          Baseline Model       |          Random Intercept Model
#                   --------------------------- | ------------------------------------
#                   | AIC | R²-cond | R²-marg   | AIC | R²-cond | R²-marg | Coeff [CI]
# -------------------------------------------------------------------------------------
#           | Dset1 |
# Predictor | Dset2 |
#           | Dset3 |


# Helper Functions that grab the data from the different result csv files and put them together into a dataset for
# the paper

# for the null models and baseline models
# inputs are the baseline task files as well as the random intercept only (null model) task files (only the diag files
# are needed, not the coefficient files)
def create_null_baseline_table(null_df, baseline_df):
    # results dict
    null_baseline_table = {}

    # get the AIC; R2_conditional, R2_marginal and ICC from the null model and the baseline model

    # loop every target and every dataset
    targets = null_df["dv"].unique()
    dsets = null_df["dframe"].unique()

    for target in targets:
        for dset in dsets:

            null_baseline_table[(target, name_change_dict[dset])] = {}

            # get the target, dset subset from the dataframes
            null_diag = null_df.loc[(null_df["dv"] == target) & (null_df["dframe"] == dset)]
            baseline_diag = baseline_df.loc[(baseline_df["dv"] == target) & (baseline_df["dframe"] == dset)]

            # get the AIC, R2_conditionaö. R_2 marginal and ICC values for the null model and baseline model
            # null model
            null_baseline_table[(target, name_change_dict[dset])][("Null Model", "AIC")] = null_diag["AIC"].values[0]
            null_baseline_table[(target, name_change_dict[dset])][("Null Model", "R²-cond")] = \
            null_diag["R2_conditional"].values[0]
            null_baseline_table[(target, name_change_dict[dset])][("Null Model", "R²-marg")] = \
            null_diag["R2_marginal"].values[0]
            null_baseline_table[(target, name_change_dict[dset])][("Null Model", "ICC")] = null_diag["ICC"].values[0]

            # baseline model
            null_baseline_table[(target, name_change_dict[dset])][("Baseline Model", "AIC")] = \
            baseline_diag["AIC"].values[0]
            null_baseline_table[(target, name_change_dict[dset])][("Baseline Model", "R²-cond")] = \
            baseline_diag["R2_conditional"].values[0]
            null_baseline_table[(target, name_change_dict[dset])][("Baseline Model", "R²-marg")] = \
            baseline_diag["R2_marginal"].values[0]
            null_baseline_table[(target, name_change_dict[dset])][("Baseline Model", "ICC")] = \
            baseline_diag["ICC"].values[0]


    return null_baseline_table


# for the single predictor results
# inputs are the five datasets for the random intercept model coefficient estimations, the random intercept model model
# diagnostics, the random slope model coefficient estimations, the random slope model model diagnostics
def create_single_pred_result_table(ri_coeffs_df, ri_diag_df, rs_coeffs_df, rs_diag_df):
    # results table dict
    single_pred_table = {}

    # loop every predictor for every data set and every dependent variable, isolate the desired results and save them
    sp_targets = ri_diag_df["dv"].unique()
    sp_dsets = ri_diag_df["dframe"].unique()
    sp_preds = ri_diag_df["iv"].unique()

    # Loop all combinations to fill the single pred_table in order to create the results dataframe
    for target in sp_targets:
        single_pred_table[target] = {}
        for pred in sp_preds:
            for dset in sp_dsets:
                # extract the relevant info from the result tables and save them in the dictionary

                # first check if the target, pred, dset combination exists
                # to do so, extract the combination from the random intercept model and get the length (should be
                # greater than 0)
                ri_diag = ri_diag_df.loc[(ri_diag_df["dv"] == target) &
                                             (ri_diag_df["iv"] == pred) &
                                             (ri_diag_df["dframe"] == dset)]

                if len(ri_diag) > 0:
                    pred_name = name_change_dict[pred]
                    dset_name = name_change_dict[dset]
                    single_pred_table[target][(pred_name, dset_name)] = {}

                    # Random Intercept Model Diagnostics (AIC, R²-cond, R²-marg, ICC)
                    single_pred_table[target][(pred_name, dset_name)][('Rand. Intercept Model', 'AIC')] = \
                        ri_diag['AIC'].values[0]
                    single_pred_table[target][(pred_name, dset_name)][('Rand. Intercept Model', 'R²-cond')] = \
                        ri_diag['R2_conditional'].values[0]
                    single_pred_table[target][(pred_name, dset_name)][('Rand. Intercept Model', 'R²-marg')] = \
                        ri_diag['R2_marginal'].values[0]
                    single_pred_table[target][(pred_name, dset_name)][('Rand. Intercept Model', 'ICC')] = \
                        ri_diag['ICC'].values[0]

                    # Random Intercept Model Coeffs
                    ri_coeffs = ri_coeffs_df.loc[(ri_coeffs_df["dv"] == target) &
                                                 (ri_coeffs_df["iv"] == pred) &
                                                 (ri_coeffs_df["dframe"] == dset) &
                                                 (ri_coeffs_df["term"] == pred)]

                    single_pred_table[target][(pred_name, dset_name)][('Rand. Intercept Model', 'Fixed Effect Est.')] = str(
                        np.round(ri_coeffs["estimate"].values[0], 2)) + "\n[" \
                                                                           + str(
                        np.round(ri_coeffs["conf.low"].values[0], 2)) + ", " + str(
                        np.round(ri_coeffs["conf.high"].values[0], 2)) + "]"
                    # single_pred_table[target][(pred, dset)]["RI_FE_Conf.low"] = ri_coeffs["conf.low"].values[0]
                    # single_pred_table[target][(pred, dset)]["RI_FE_Conf.high"] = ri_coeffs["conf.high"].values[0]

                    # Random Slope Model Diagnostics (AIC, R²-cond, R²-marg, ICC)
                    rs_diag = rs_diag_df.loc[(rs_diag_df["dv"] == target) &
                                             (rs_diag_df["iv"] == pred) &
                                             (rs_diag_df["dframe"] == dset)]

                    single_pred_table[target][(pred_name, dset_name)][('Rand. Intercept & Slope Model', 'AIC')] = \
                        rs_diag['AIC'].values[0]
                    single_pred_table[target][(pred_name, dset_name)][('Rand. Intercept & Slope Model', 'R²-cond')] = \
                        rs_diag['R2_conditional'].values[0]
                    single_pred_table[target][(pred_name, dset_name)][('Rand. Intercept & Slope Model', 'R²-marg')] = \
                        rs_diag['R2_marginal'].values[0]
                    single_pred_table[target][(pred_name, dset_name)][('Rand. Intercept & Slope Model', 'ICC')] = \
                        rs_diag['ICC'].values[0]

                    # Random Slope Model Coeff

                    # fixed term
                    rs_coeffs = rs_coeffs_df.loc[(rs_coeffs_df["dv"] == target) &
                                                 (rs_coeffs_df["iv"] == pred) &
                                                 (rs_coeffs_df["dframe"] == dset) &
                                                 (rs_coeffs_df["term"] == pred)]

                    single_pred_table[target][(pred_name, dset_name)][('Rand. Intercept & Slope Model', 'Fixed Effect Est.')] = str(
                        np.round(rs_coeffs["estimate"].values[0], 2)) + "\n[" \
                                                                           + str(
                        np.round(rs_coeffs["conf.low"].values[0], 2)) + ", " + str(
                        np.round(rs_coeffs["conf.high"].values[0], 2)) + "]"
                    # single_pred_table[target][(pred, dset)]["RS_FE_Conf.low"] = rs_coeffs["conf.low"].values[0]
                    # single_pred_table[target][(pred, dset)]["RS_FE_Conf.high"] = rs_coeffs["conf.high"].values[0]

                    # random term
                    rs_rand_coeff = rs_coeffs_df.loc[(rs_coeffs_df["dv"] == target) &
                                                     (rs_coeffs_df["iv"] == pred) &
                                                     (rs_coeffs_df["dframe"] == dset) &
                                                     (rs_coeffs_df["term"] == 'sd__' + pred)]

                    single_pred_table[target][(pred_name, dset_name)][('Rand. Intercept & Slope Model', 'Rand. Effect Est.')] = rs_rand_coeff["estimate"].values[0]

    return single_pred_table


# for the interaction effect results
# inputs are the five datasets for the random intercept model coefficient estimations, the random intercept model model
# diagnostics, the random slope model coefficient estimations, the random slope model model diagnostics
def create_interaction_results_table(ri_coeffs_df, ri_diag_df, rs_coeffs_df, rs_diag_df):
    # setup a dictionary to store the grabbed results in order to convert it to the results table
    ie_table = {}

    # get all interaction combinations that will be looped
    ie_targets = ri_diag_df["dv"].unique()
    ie_dsets = ri_diag_df["dframe"].unique()
    ie_combs = ri_diag_df.loc[:, ["iv1", "iv2"]].drop_duplicates().values

    # Loop all combinations to fill the single pred_table in order to create the results dataframe
    for target in ie_targets:
        ie_table[target] = {}
        for int_pair in ie_combs:
            for dset in ie_dsets:
                # extract the relevant info from the result tables and save them in the dictionary

                # first check if the target, pred, dset combination exists
                # to do so, extract the combination from the random intercept model and get the length (should be
                # greater than 0)
                # Random Intercept Model R²-scores
                ri_diag = ri_diag_df.loc[(ri_diag_df["dv"] == target) &
                                         (ri_diag_df["iv1"] == int_pair[0]) &
                                         (ri_diag_df["iv2"] == int_pair[1]) &
                                         (ri_diag_df["dframe"] == dset)]

                if len(ri_diag) > 0:
                    ie_name = name_change_dict[int_pair[0]] + "\nX\n" + name_change_dict[int_pair[1]]
                    dset_name = name_change_dict[dset]
                    ie_table[target][(ie_name, dset_name)] = {}

                    # Random Intercept Model Diagnostics (AIC, R²-cond, R²-marg, ICC)
                    ie_table[target][(ie_name, dset_name)][('Rand. Intercept Model', 'AIC')] = \
                        ri_diag['AIC'].values[0]
                    ie_table[target][(ie_name, dset_name)][('Rand. Intercept Model', 'R²-cond')] = \
                        ri_diag['R2_conditional'].values[0]
                    ie_table[target][(ie_name, dset_name)][('Rand. Intercept Model', 'R²-marg')] = \
                        ri_diag['R2_marginal'].values[0]
                    ie_table[target][(ie_name, dset_name)][('Rand. Intercept Model', 'ICC')] = \
                        ri_diag['ICC'].values[0]

                    # Random Intercept Model Coeffs
                    ri_coeffs = ri_coeffs_df.loc[(ri_coeffs_df["dv"] == target) &
                                                 (ri_coeffs_df["iv1"] == int_pair[0]) &
                                                 (ri_coeffs_df["iv2"] == int_pair[1]) &
                                                 (ri_coeffs_df["dframe"] == dset) &
                                                 (ri_coeffs_df["term"] == int_pair[0] + ":" + int_pair[1])]

                    ie_table[target][(ie_name, dset_name)][('Rand. Intercept Model', 'Fixed Effect Est.')] = \
                        str(np.round(ri_coeffs["estimate"].values[0], 2)) + "\n[" \
                        + str(np.round(ri_coeffs["conf.low"].values[0], 2)) + ", " + str(
                            np.round(ri_coeffs["conf.high"].values[0], 2)) + "]"
                    # ie_table[target][(int_pair[0] + "*" + int_pair[1], dset)]["RI_FE_Conf.low"] = ri_coeffs["conf.low"].values[0]
                    # ie_table[target][(int_pair[0] + "*" + int_pair[1], dset)]["RI_FE_Conf.high"] = ri_coeffs["conf.high"].values[0]

                    # Random Slope Model R²-scores
                    rs_diag = rs_diag_df.loc[(rs_diag_df["dv"] == target) &
                                             (rs_diag_df["iv1"] == int_pair[0]) &
                                             (rs_diag_df["iv2"] == int_pair[1]) &
                                             (rs_diag_df["dframe"] == dset)]

                    ie_table[target][(ie_name, dset_name)][('Rand. Intercept & Slope Model', 'AIC')] = \
                        rs_diag['AIC'].values[0]
                    ie_table[target][(ie_name, dset_name)][('Rand. Intercept & Slope Model', 'R²-cond')] = \
                        rs_diag['R2_conditional'].values[0]
                    ie_table[target][(ie_name, dset_name)][('Rand. Intercept & Slope Model', 'R²-marg')] = \
                        rs_diag['R2_marginal'].values[0]
                    ie_table[target][(ie_name, dset_name)][('Rand. Intercept & Slope Model', 'ICC')] = \
                        rs_diag['ICC'].values[0]

                    # Random Slope Model Coeff

                    # fixed term
                    rs_coeffs = rs_coeffs_df.loc[(rs_coeffs_df["dv"] == target) &
                                                 (rs_coeffs_df["iv1"] == int_pair[0]) &
                                                 (rs_coeffs_df["iv2"] == int_pair[1]) &
                                                 (rs_coeffs_df["dframe"] == dset) &
                                                 (rs_coeffs_df["term"] == int_pair[0] + ":" + int_pair[1])]

                    ie_table[target][(ie_name, dset_name)][('Rand. Intercept & Slope Model', 'Fixed Effect Est.')] = \
                        str(np.round(rs_coeffs["estimate"].values[0], 2)) + "\n[" \
                        + str(np.round(rs_coeffs["conf.low"].values[0], 2)) + ", " + str(
                            np.round(rs_coeffs["conf.high"].values[0], 2)) + "]"
                    # ie_table[target][(int_pair[0] + "*" + int_pair[1], dset)]["RS_FE_Conf.low"] = rs_coeffs["conf.low"].values[0]
                    # ie_table[target][(int_pair[0] + "*" + int_pair[1], dset)]["RS_FE_Conf.high"] = rs_coeffs["conf.high"].values[0]

                    # random term
                    rs_rand_coeff = rs_coeffs_df.loc[(rs_coeffs_df["dv"] == target) &
                                                     (rs_coeffs_df["iv1"] == int_pair[0]) &
                                                     (rs_coeffs_df["iv2"] == int_pair[1]) &
                                                     (rs_coeffs_df["dframe"] == dset) &
                                                     (rs_coeffs_df["term"] == 'sd__' + int_pair[0] + ":" + int_pair[1])]

                    ie_table[target][(ie_name, dset_name)][('Rand. Intercept & Slope Model', 'Rand. Effect Est.')] = \
                        rs_rand_coeff["estimate"].values[0]

    return ie_table


# %%

# -----------
# Task Data -
# -----------

# Null-Model Results and Baseline Results
# ---------------------------------------

# import the datasets
task_null_mod_diag = pd.read_csv("Mouse_Task_Results/Mixed_Models/Random_Intercept/Task_results_ri_diag.csv")
task_baseline_mod_diag = pd.read_csv("Mouse_Task_Results/Mixed_Models/Baseline/Task_results_baseline_diag.csv")

#%%

# get the results table
task_null_baseline_result_table = create_null_baseline_table(task_null_mod_diag, task_baseline_mod_diag)

#%%


# Single Predictor Model results
# ------------------------------

# import all results from the csv file (there are 5 result files)
task_sp_fe_coeffs = pd.read_csv("Mouse_Task_Results/Mixed_Models/Single_Predictor/Task_results_sp_fe_coeffs.csv")
task_sp_fe_diag = pd.read_csv("Mouse_Task_Results/Mixed_Models/Single_Predictor/Task_results_sp_fe_diag.csv")
task_sp_re_coeffs = pd.read_csv("Mouse_Task_Results/Mixed_Models/Single_Predictor/Task_results_sp_re_coeffs.csv")
task_sp_re_diag = pd.read_csv("Mouse_Task_Results/Mixed_Models/Single_Predictor/Task_results_sp_re_diag.csv")

# %%

# get the results table
task_single_pred_results_table = create_single_pred_result_table(task_sp_fe_coeffs, task_sp_fe_diag,
                                                                 task_sp_re_coeffs, task_sp_re_diag)

# %%

# Interaction Effect Results
# ---------------------------

task_ie_fe_coeff = pd.read_csv("Mouse_Task_Results/Mixed_Models/Interactions/Task_results_interaction_fe_coeffs.csv")
task_ie_fe_diag = pd.read_csv("Mouse_Task_Results/Mixed_Models/Interactions/Task_results_interaction_fe_diags.csv")
task_ie_re_coeff = pd.read_csv("Mouse_Task_Results/Mixed_Models/Interactions/Task_results_interaction_re_coeffs.csv")
task_ie_re_diag = pd.read_csv("Mouse_Task_Results/Mixed_Models/Interactions/Task_results_interaction_re_diags.csv")

# %%

# get the results table
task_interaction_results_table = create_interaction_results_table(task_ie_fe_coeff, task_ie_fe_diag,
                                                                  task_ie_re_coeff, task_ie_re_diag)


# %%

# -----------------------
# Free Mouse Usage Data -
# -----------------------

# Null-Model Results and Baseline Results
# ---------------------------------------

# import the datasets
free_null_mod_diag = pd.read_csv("Free_Mouse_Results/Mixed_Model_Results/Random_Intercept/Free_Mouse_results_ri_diag.csv")
free_baseline_mod_diag = pd.read_csv("Free_Mouse_Results/Mixed_Model_Results/Baseline/Free_results_baseline_diag.csv")

#%%

# get the results table
free_null_baseline_result_table = create_null_baseline_table(free_null_mod_diag, free_baseline_mod_diag)

#%%

# Single Predictor Model results
# ------------------------------

# import all results from the csv file (there are 5 result files)
free_sp_fe_coeffs = pd.read_csv(
    "Free_Mouse_Results/Mixed_Model_Results/Single_predictor/Free_Mouse_results_sp_fe_coeffs.csv")
free_sp_fe_diag = pd.read_csv(
    "Free_Mouse_Results/Mixed_Model_Results/Single_predictor/Free_Mouse_results_sp_fe_diag.csv")
free_sp_re_coeffs = pd.read_csv(
    "Free_Mouse_Results/Mixed_Model_Results/Single_predictor/Free_Mouse_results_sp_re_coeffs.csv")
free_sp_re_diag = pd.read_csv(
    "Free_Mouse_Results/Mixed_Model_Results/Single_predictor/Free_Mouse_results_sp_re_diag.csv")

# %%

# get the results table
free_single_pred_results_table = create_single_pred_result_table(free_sp_fe_coeffs, free_sp_fe_diag,
                                                                 free_sp_re_coeffs, free_sp_re_diag)

# %%

# Interaction Effect Results
# ---------------------------

free_ie_fe_coeff = pd.read_csv(
    "Free_Mouse_Results/Mixed_Model_Results/Interactions/Free_Mouse_results_interaction_fe_coeffs.csv")
free_ie_fe_diag = pd.read_csv(
    "Free_Mouse_Results/Mixed_Model_Results/Interactions/Free_Mouse_results_interaction_fe_diags.csv")
free_ie_re_coeff = pd.read_csv(
    "Free_Mouse_Results/Mixed_Model_Results/Interactions/Free_Mouse_results_interaction_re_coeffs.csv")
free_ie_re_diag = pd.read_csv(
    "Free_Mouse_Results/Mixed_Model_Results/Interactions/Free_Mouse_results_interaction_re_diags.csv")

# %%

# get the results table
free_interaction_results_table = create_interaction_results_table(free_ie_fe_coeff, free_ie_fe_diag,
                                                                  free_ie_re_coeff, free_ie_re_diag)


#%%

# Get some descriptive stats about the models, because manually reading them from the tables is tideous
# -----------------------------------------------------------------------------------------------------

# helper function to get some descriptive stats about the null-model/baseline-model comparison
def null_baseline_comparison(baseline_results, target):

    print(f'Null-Baseline Comparions for target {target}')

    # first create dataframes for the baseline results and model results using the specified target
    baseline_df = pd.DataFrame.from_dict(baseline_results, orient='index').rename_axis(["target", "dset"]).round(2).loc[
                  target, :]

    # drop duplicates (because standardization doesnt matter)
    baseline_df = baseline_df.drop_duplicates(subset=[("Null Model", "AIC")])

    # get the difference scores between null modell and baseline model
    baseline_df[("Comp", "AIC")] = baseline_df[("Baseline Model", "AIC")] - baseline_df[("Null Model", "AIC")]
    baseline_df[("Comp", "R²-cond")] = baseline_df[("Baseline Model", "R²-cond")] - baseline_df[("Null Model", "R²-cond")]

    if len(baseline_df) > 1:
        print(f"Mean delta AIC: {baseline_df[('Comp', 'AIC')].mean()}")
        print(f"SD delta AIC: {baseline_df[('Comp', 'AIC')].std()}")
        print(f"Mean delta R²-cond: {baseline_df[('Comp', 'R²-cond')].mean()}")
        print(f"SD delta R²-cond: {baseline_df[('Comp', 'R²-cond')].std()}")
    else:
        print(f"Delta AIC: {baseline_df[('Comp', 'AIC')]}")
        print(f"Delta R²-cond: {baseline_df[('Comp', 'R²-cond')]}")

# bad coded helper function to get some descriptive information about the results for the paper
# bad coded because some things are redundant and it might be better to get the desc. stats in differen step, with
# a different approach (the following was the straightforward, short cut solution)
# e.g. which of the competing models (baseline vs. random intercept vs. random intercept & slope) has the "best fit"
def describe_results(baseline_results, model_results, target):

    print(f"Get some descriptive results for target: {target}")
    # first create dataframes for the baseline results and model results using the specified target
    baseline_df = pd.DataFrame.from_dict(baseline_results, orient='index').rename_axis(["target", "dset"]).round(2).loc[target, :]

    model_df = pd.DataFrame.from_dict(model_results[target], orient='index').rename_axis(["pred", "dset"]).round(2)

    # merge both datasets
    merged_df = model_df.join(baseline_df, how="inner")

    print(f"Total number of models: {len(merged_df)}")

    # create a column that contains the model name of the model with the lowest AIC value = best fit
    merged_df[("Stats", "AIC_max")] = merged_df.loc[:, merged_df.columns.get_level_values(1) == "AIC"].idxmin(axis=1).str[0]

    # print info about which model was the best
    best_aic_models = merged_df[('Stats', 'AIC_max')].value_counts()
    print(f"Best AIC values by models:\n{best_aic_models}")
    print(f"Percentage of better than baseline models: {(best_aic_models['Rand. Intercept & Slope Model'] + best_aic_models['Rand. Intercept Model'])/ len(merged_df) }")
    print(f"Percentage of Ran. Intercept Model: {best_aic_models['Rand. Intercept Model'] / (best_aic_models['Rand. Intercept Model'] + best_aic_models['Rand. Intercept & Slope Model'])}")

    # get all best random intercept models
    best_ri_models = merged_df[merged_df[('Stats', 'AIC_max')] == "Rand. Intercept Model"].loc[:, "Rand. Intercept Model"]
    # get all best random intercept & slope models
    best_slope_models = merged_df[merged_df[('Stats', 'AIC_max')].str.contains("Slope")].loc[:, "Rand. Intercept & Slope Model"]

    # get the variables, which have at least one model. which is better than the baseline model
    sig_ri_vars = list(best_ri_models.index.get_level_values(0).unique())
    sig_slope_vars = list(best_slope_models.index.get_level_values(0).unique())
    sig_variables = np.unique(sig_ri_vars + sig_slope_vars)
    print(f"Sig variables for the random intercept model: {len(sig_ri_vars)}")
    # print(sig_ri_vars)
    print(f"Sig variables for the random slope model: {len(sig_slope_vars)}")
    # print(sig_slope_vars)
    print(f"Total significant variables: {len(sig_variables)}")
    # print(f"The variables are: {sig_variables}")
    print(f"Total number of variables: {len(model_df.index.get_level_values(0).unique())}")

    # get descriptive stats about the model diagnostic criteria and model estimates of the "best" models
    # first create separate columns for the fixed effect estimates and the confidence interval bounds (was a string var)
    # random intercept models
    best_ri_models[["Est.", "Conf.l", "Conf.h"]] = best_ri_models.loc[:, "Fixed Effect Est."].str.split(
        expand=True).replace(to_replace=r'[]|,|[]', value='', regex=True).apply(pd.to_numeric)
    # add a column that indicates if the CI contains 0
    best_ri_models["CI_includes_0"] = (0 >= best_ri_models.loc[:, 'Conf.l']) & (0 <= best_ri_models.loc[:, 'Conf.h'])
    # random slope models
    best_slope_models[["Est.", "Conf.l", "Conf.h"]] = best_slope_models.loc[:, "Fixed Effect Est."].str.split(
        expand=True).replace(to_replace=r'[]|,|[]', value='', regex=True).apply(pd.to_numeric)
    # add a column that indicates if the CI contains 0
    best_slope_models["CI_includes_0"] = (0 >= best_slope_models.loc[:, 'Conf.l']) & \
                                         (0 <= best_slope_models.loc[:, 'Conf.h'])

    print(f"Number of CIs that contain 0 for the random intercept model:\n{best_ri_models['CI_includes_0'].value_counts()}")
    print(f"Number of CIs that contain 0 for the random slope model:\n{best_slope_models['CI_includes_0'].value_counts()}")

    # stack the best ri_model and the best slope_model and get infos about the range of coefficient estimates
    stacked = pd.concat([best_ri_models, best_slope_models])
    print(f"Range of R²-cond: {stacked['R²-cond'].min()}, {stacked['R²-cond'].max()}")
    print(f"Range of R²-marg: {stacked['R²-marg'].min()}, {stacked['R²-marg'].max()}")
    print(f"Range of fixed-effect: {stacked['Est.'].min()}, {stacked['Est.'].max()}")
    print(f"Largest fixed effect for predictor and dataset: {stacked['Est.'].abs().idxmax()}")
    print(f"Largest fixed effect: {stacked.loc[stacked['Est.'].abs().idxmax(), ['Est.', 'Conf.l', 'Conf.h']]}")
    print(f"Range of random effect: {stacked['Rand. Effect Est.'].min()}, {stacked['Rand. Effect Est.'].max()}")
    print(f"Largest random effect for predictor and dataset: {best_slope_models['Rand. Effect Est.'].idxmax()}")

    # compare the model estimates between the different datasets for each dependent variable with multiple datasets

    # helper function to get confidence interval overlaps
    def _get_ci_overlaps(df):
        # get the confidence intervals of the fixed effect estimates
        df[["Est.", "Conf.l", "Conf.h"]] = df.loc[:, "Fixed Effect Est."].str.split(
            expand=True).replace(to_replace=r'[]|,|[]', value='', regex=True).apply(pd.to_numeric)
        # following code from:
        # https://stackoverflow.com/questions/66343650/how-to-efficiently-find-overlapping-intervals
        l1 = df['Conf.l'].to_numpy()
        h1 = df['Conf.h'].to_numpy()
        l2 = l1[:, None]
        h2 = h1[:, None]
        # Check for overlap
        # mask is an n * n matrix indicating if interval i overlaps with interval j
        mask = (l1 < h2) & (h1 > l2)
        # If interval i overlaps intervla j then j also overlaps i. We only want to get
        # one of the two pairs. Hence the `triu` (triangle, upper)
        # Every interval also overlaps itself and we don't want that either. Hence the k=1
        overlaps = np.triu(mask, k=1).nonzero()

        return overlaps

    non_overlapping_cis_ri_mod = []
    non_overlapping_cis_slope_mod = []
    random_effect_variation = {}
    # loop all predictors
    for pred in model_df.index.get_level_values(0).unique():
        # loc the datasets for the predictor (random intercept model & random slope model)
        pred_ri_df = model_df.loc[pred, "Rand. Intercept Model"]
        pred_slope_df = model_df.loc[pred, "Rand. Intercept & Slope Model"]
        # can only compare the preprocessing datasets if the predictors has more than one
        if len(pred_ri_df) > 1:
            # get ci overlaps
            ri_model_overlaps = _get_ci_overlaps(pred_ri_df)
            slope_model_overlaps = _get_ci_overlaps(pred_slope_df)
            # check if the number of overlaps is not equal to the number of all overlap combinations
            if len(ri_model_overlaps[0]) != sum([i for i in range(len(pred_ri_df))]):
                non_overlapping_cis_ri_mod.append(pred)
                print(f"Range of estimates for {pred}: {pred_ri_df['Est.'].min()}, {pred_ri_df['Est.'].max()}")
            # check the same for the random slope model
            if len(slope_model_overlaps[0]) != sum([i for i in range(len(pred_ri_df))]):
                non_overlapping_cis_slope_mod.append(pred)
                print(f"Range of estimates for {pred}: {pred_slope_df['Est.'].min()}, {pred_slope_df['Est.'].max()}")

            # get the standard deviation of the random effect coefficient estimates between the datasets
            random_effect_variation[pred] = pred_slope_df['Rand. Effect Est.'].std()
            # print(pred, pred_slope_df['Rand. Effect Est.'].std())

    print(f"Number of predictors with multiple datasets: {len(random_effect_variation)}")
    print(f"CI non-overlapping in random intercept model for variables: {non_overlapping_cis_ri_mod}")
    print(f"CI non-overlapping in random slope model for variables: {non_overlapping_cis_slope_mod}")

    # convert the dict into a df, and get some stats
    rand_eff_var_df = pd.Series(random_effect_variation).T
    print(f"Mean random effect variation between datasets: {rand_eff_var_df.mean()}")
    print(f"Min random effect variation between datasets: {rand_eff_var_df.min()}")
    print(f"Max random effect variation between datasets: {rand_eff_var_df.max()}")
    print(f"Max random effect variation between datasets: {rand_eff_var_df.idxmax()}")

    # return best ri models and best slope models
    return best_ri_models, best_slope_models


#%%

# get the descriptive stats for the single predictor model and the interaction model, per target variable (arousal,
# valence, stress) and for the mouse task data and the free-mouse data

# mouse task data
# ...............

# baseline comparison
null_baseline_comparison(task_null_baseline_result_table, 'arousal')
null_baseline_comparison(task_null_baseline_result_table, 'valence')
null_baseline_comparison(task_null_baseline_result_table, 'stress')

#%%

# single pred models
print("Get desc result stats for the mouse task single predictor models and target arousal")
arousal_intercept, arousal_slope = describe_results(task_null_baseline_result_table, task_single_pred_results_table, 'arousal')
arousal_intercept_desc, arousal_slope_desc = arousal_intercept.describe(), arousal_slope.describe()

print("\nGet desc result stats for the mouse task single predictor models and target valence")
valence_intercept, valence_slope = describe_results(task_null_baseline_result_table, task_single_pred_results_table, 'valence')
valence_intercept_desc, valence_slope_desc = valence_intercept.describe(), valence_slope.describe()

print("\nGet desc result stats for the mouse task single predictor models and target stress")
stress_intercept, stress_slope = describe_results(task_null_baseline_result_table, task_single_pred_results_table, 'stress')
stress_intercept_desc, stress_slope_desc = stress_intercept.describe(), stress_slope.describe()

#%%
# interaction models

print("Get desc result stats for the mouse task interaction models and target arousal")
arousal_intercept, arousal_slope = describe_results(task_null_baseline_result_table, task_interaction_results_table, 'arousal')
arousal_intercept_desc, arousal_slope_desc = arousal_intercept.describe(), arousal_slope.describe()

print("\nGet desc result stats for the mouse task interaction models and target valence")
valence_intercept, valence_slope = describe_results(task_null_baseline_result_table, task_interaction_results_table, 'valence')
valence_intercept_desc, valence_slope_desc = valence_intercept.describe(), valence_slope.describe()

print("\nGet desc result stats for the mouse task interaction models and target stress")
stress_intercept, stress_slope = describe_results(task_null_baseline_result_table, task_interaction_results_table, 'stress')
stress_intercept_desc, stress_slope_desc = stress_intercept.describe(), stress_slope.describe()

#%%

# free mouse data
# ...............

# baseline comparison
null_baseline_comparison(free_null_baseline_result_table, 'arousal')
null_baseline_comparison(free_null_baseline_result_table, 'valence')
null_baseline_comparison(free_null_baseline_result_table, 'stress')


#%%

# single pred models

print("Get desc result stats for the free-mouse single predictor models and target arousal")
arousal_intercept, arousal_slope = describe_results(free_null_baseline_result_table, free_single_pred_results_table, 'arousal')
arousal_intercept_desc, arousal_slope_desc = arousal_intercept.describe(), arousal_slope.describe()

#%%

print("\nGet desc result stats for the free-mouse single predictor models and target valence")
valence_intercept, valence_slope = describe_results(free_null_baseline_result_table, free_single_pred_results_table, 'valence')
valence_intercept_desc, valence_slope_desc = valence_intercept.describe(), valence_slope.describe()

#%%

print("\nGet desc result stats for the free-mouse single predictor models and target stress")
stress_intercept, stress_slope = describe_results(free_null_baseline_result_table, free_single_pred_results_table, 'stress')
stress_intercept_desc, stress_slope_desc = stress_intercept.describe(), stress_slope.describe()

#%%

# interaction models

print("Get desc result stats for the mouse task interaction models and target arousal")
arousal_intercept, arousal_slope = describe_results(free_null_baseline_result_table, free_interaction_results_table, 'arousal')
arousal_intercept_desc, arousal_slope_desc = arousal_intercept.describe(), arousal_slope.describe()

#%%

print("\nGet desc result stats for the mouse task interaction models and target valence")
valence_intercept, valence_slope = describe_results(free_null_baseline_result_table, free_interaction_results_table, 'valence')
valence_intercept_desc, valence_slope_desc = valence_intercept.describe(), valence_slope.describe()

#%%

print("\nGet desc result stats for the mouse task interaction models and target stress")
stress_intercept, stress_slope = describe_results(free_null_baseline_result_table, free_interaction_results_table, 'stress')
stress_intercept_desc, stress_slope_desc = stress_intercept.describe(), stress_slope.describe()

#%%

############################
# Machine Learning Results #
############################

# create a new dataframe with the following structure

#                               Baseline      Full Mod
#                            -------------  -----------
#           Train/Test Shape |  R² | MAE |  | R² | MAE |
# -------------------------------------------------------
#  Dset1 |
#  Dset2 |
#  Dset3 |


# helper function to extract the relevant results from the results data set
def ml_results_table(result_files):
    result_table = {}

    # get the target variables to create separate tables per target
    targets = ["arousal", "valence", "stress"]

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

                if target != "stress":
                    # add the baseline results
                    result_table[target][dset_name][("Baseline Model", "num preds.")] = len(
                        result_files[result]['predictors']['baseline'])
                    result_table[target][dset_name][("Baseline Model", "R²-score")] = result_files[result]['baseline_results']['scores']['r2']
                    result_table[target][dset_name][("Baseline Model", "MAE")] = result_files[result]['baseline_results']['scores']['mae']
                    # add the full model results
                    result_table[target][dset_name][("Full Model", 'num. preds.')] = len(
                        result_files[result]['predictors']['full_model'])
                    result_table[target][dset_name][("Full Model", "R²-score")] = result_files[result]['full_model_results']['scores']['r2']
                    result_table[target][dset_name][("Full Model", "MAE")] = result_files[result]['full_model_results']['scores'][
                        'mae']
                    # add the mouse only results
                    result_table[target][dset_name][("Mouse Model", "num preds.")] = len(
                        result_files[result]['predictors']['mouse_only'])
                    result_table[target][dset_name][("Mouse Model", "R²-score")] = \
                        result_files[result]['mouse_only_model_results']['scores']['r2']
                    result_table[target][dset_name][("Mouse Model", "MAE")] = \
                        result_files[result]['mouse_only_model_results']['scores']['mae']
                else:
                    # add the baseline results
                    result_table[target][dset_name][("Baseline Model", "num preds.")] = len(
                        result_files[result]['predictors']['baseline'])
                    result_table[target][dset_name][("Baseline Model", "Acc")] = result_files[result]['baseline_results']['scores']['acc']
                    result_table[target][dset_name][("Baseline Model", "bal. Acc")] = result_files[result]['baseline_results']['scores'][
                        'b_acc']
                    result_table[target][dset_name][("Baseline Model", "f1")] = result_files[result]['baseline_results']['scores']['f1']
                    result_table[target][dset_name][("Baseline Model", "conf.Mat.")] = result_files[result]['baseline_results']['scores']['cm']
                    # add the full model results
                    result_table[target][dset_name][("Full Model", 'num. preds.')] = len(
                        result_files[result]['predictors']['full_model'])
                    result_table[target][dset_name][("Full Model", 'Acc')] = result_files[result]['full_model_results']['scores'][
                        'acc']
                    result_table[target][dset_name][("Full Model", 'bal. Acc')] = result_files[result]['full_model_results']['scores'][
                        'b_acc']
                    result_table[target][dset_name][("Full Model", 'f1')] = result_files[result]['full_model_results']['scores']['f1']
                    result_table[target][dset_name][("Full Model", 'conf.Mat.')] = result_files[result]['full_model_results']['scores']['cm']
                    # add the mouse only results
                    result_table[target][dset_name][("Mouse Model", "num preds.")] = len(
                        result_files[result]['predictors']['mouse_only'])
                    result_table[target][dset_name][("Mouse Model", "Acc")] = \
                        result_files[result]['mouse_only_model_results']['scores']['acc']
                    result_table[target][dset_name][("Mouse Model", "bal. Acc")] = \
                        result_files[result]['mouse_only_model_results']['scores']['b_acc']
                    result_table[target][dset_name][("Mouse Model", "f1")] = \
                        result_files[result]['mouse_only_model_results']['scores']['f1']
                    result_table[target][dset_name][("Mouse Model", "conf.Mat.")] = \
                        result_files[result]['mouse_only_model_results']['scores']['cm']

    return result_table


# %%

# Mouse Usage Task
# ----------------

with open("Mouse_Task_Results/Machine_Learning/mouse_task_ML_results.json", "r") as f:
    task_ml_results = json.loads(json.load(f))

# %%

task_ml_result_table = ml_results_table(task_ml_results)

# %%

# Free Mouse Usage
# -----------------

with open("Free_Mouse_Results/ML_Results/Free_Mouse_ML_results.json", "r") as g:
    free_ml_results = json.loads(json.load(g))

# %%

free_ml_results_table = ml_results_table(free_ml_results)


# %%

############################
# Create Tables per Target #
############################

# create the result tables and save them as csv files (they will need further manual processing in order to fit the
# APA format)


# helper function to save the output table as csv files, the baseline table and single predictor/interaction table
# need separate helper functions
def save_null_baseline_output(table_dict, filename):

    # convert the null model & baseline model dictionary to a dataframe
    df = pd.DataFrame.from_dict(table_dict, orient='index').rename_axis(["target", "dset"]).round(2)
    # save it as an excel file
    df.to_excel(filename + '.xlsx')


def save_mixed_model_output(table_dict, filename):
    # create an excel file with separate sheets that contain the results for each target variable
    with pd.ExcelWriter(filename + ".xlsx") as writer:
        # the table dict contains a separate dictionary for each target variable (alternative would be to create one
        # large multiindex dataframe)
        for target in table_dict:
            # convert the target table dict into a dataframe
            table_df = pd.DataFrame.from_dict(table_dict[target], orient='index').rename_axis(["pred", "dset"]).round(2)
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
# save_null_baseline_output(task_null_baseline_result_table, "task_null_baseline_res_table")
# save_mixed_model_output(task_single_pred_results_table, "task_single_pred_res_table")
# save_mixed_model_output(task_interaction_results_table, "task_interaction_res_table")
# save_null_baseline_output(free_null_baseline_result_table, "free_null_baseline_res_table")
# save_mixed_model_output(free_single_pred_results_table, "free_single_pred_res_table")
# save_mixed_model_output(free_interaction_results_table, "free_interaction_res_table")

# save_ml_output(task_ml_result_table, 'task_ml_res_table')
# save_ml_output(free_ml_results_table, 'free_ml_res_table')
