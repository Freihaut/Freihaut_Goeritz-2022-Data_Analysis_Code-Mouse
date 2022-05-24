'
Validate the mouse feature calculation procedures against the mousetrap R package mouse feature
calculation procedures with sample data (1 data row per participant)
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'

# setup
library(mousetrap)
library(jsonlite)
library(dplyr)
# to show images in a separate window
options(device='windows')

# import the raw json mousetrap validation dataset
raw_validation_data <- fromJSON("Mousetrap_Validation_Data.json")

# get the calculated python mouse features
py_features <- raw_validation_data[["Py_Mouse_Features"]]

# get the raw mouse data as a dataframe in order to import it as a mousetrap data object
raw_mouse_data <- dplyr::bind_rows(raw_validation_data["Raw_Data"])
# the columns of the dataframe are lists so unlist them
raw_mouse_data <- as.data.frame(lapply(raw_mouse_data, unlist))

# only select the relevant features
raw_mouse_data <- dplyr::select(raw_mouse_data, c(cN, x, y, t))

# create the mousetrap object, the data is split by trials (mousetrap doesnt like data with only one trial -> entire task)
mt_data <- mt_import_long(
  raw_data = raw_mouse_data,
  xpos_label = 'x',
  ypos_label = 'y',
  timestamps_label = 't',
  mt_id_label = "cN"
)

# plot the mouse trajectory of the mouse task to check that the dataset contains valid data and that the mousetrap
# import procedure was correct
mt_plot(mt_data)

# interpolate the movement datapoints into equal time steps, the mousetrap package includes the last timestamp even
# if it does not fit into the interpolation step. This is not the case for the python interpolation, therefore set
# the option to false
mt_data <- mt_resample(mt_data, step_size = 15, exact_last_timestamp = F)

# use different mouse trap calculation functions to calculate distance, speed, accerleration, x-&y-flips, sample entropy
# and the angle (use the resampled, i.e. interpolated mouse data
mt_data <- mt_derivatives(mt_data, use = 'rs_trajectories')
mt_data <- mt_measures(mt_data, use = 'rs_trajectories')
# mt_data <- mt_sample_entropy(mt_data, use = 'rs_trajectories')
mt_data <- mt_angles(mt_data, use = 'rs_trajectories', unit = 'degree')

# aggregate the features over the trial -> mean trial py features, calculates the averages for the mt_measures
avg_measures <- mt_aggregate(mt_data)

# the distance, velocity, acceleration and angle data are saved as individual datapoints in the trajectories matrix,
# to calculate the means, they must be processed first

# extract the velocity, but remove the first column
# the mousetrap package fills the first column with a 0 value, which needs to be removed for calculating the mean
# the 0 is added because to calculate the speed, 2 mouse positions are needed, so the velocities have one datapoint
# less than the mouse position values
mean_vel <- mt_data[3]$rs_trajectories[,, 'vel'][,-1]
# calculate the trial mean
mean_vel <- mean(rowMeans(mean_vel * 1000, na.rm = TRUE))

# do the same for acceleration
mean_acc <- mt_data[3]$rs_trajectories[,, 'acc'][,-1]
# take the absolute value
mean_acc <- mean(rowMeans(abs(mean_acc * 1000), na.rm = TRUE))

# for the angle, the first column is already a nan value and doesnt need to be removed
mean_angle <- mt_data[3]$rs_trajectories[,, 'angle_p']
# the angles need to be reversed by subtracting 180 degrees
mean_angle <-180 - mean(rowMeans(mean_angle, na.rm = TRUE))

# compare selected features between the mousetrap feature calculation procedure and the py feature calculation procedure
feature_comparison <- data.frame(
  x_flips = c(avg_measures$xpos_flips, py_features[["trial_mean_x_flips"]]),
  y_flips = c(avg_measures$ypos_flips, py_features[["trial_mean_y_flips"]]),
  # x_entropy = c(avg_measures$sample_entropy, py_features[["trial_mean_x_entropy"]]),
  distance = c(avg_measures$total_dist, py_features[["trial_mean_total_dist"]]),
  mean_angles = c(mean_angle, py_features[["trial_mean_angle_mean"]]),
  mean_speed = c(mean_vel, py_features[["trial_mean_speed_mean"]]),
  mean_acc = c(mean_acc, py_features[["trial_mean_abs_accel_mean"]]),
  row.names = c("mousetrap_feat", "py_feat")
)

# output the feature comparison dataframe
# Differences are likely due to small calculation differences between R and Python
# The entropy values are different because in the mousetrap package, the r-threshold value is calculated using the
# standard deviation of all data across all trials (r <- 0.2 * stats::sd(diff(t(trajectories[, , dimension])), na.rm = TRUE))
# In our case, we calculated r-thresholds using the the standard deviation of each trial dataset seperately
feature_comparison