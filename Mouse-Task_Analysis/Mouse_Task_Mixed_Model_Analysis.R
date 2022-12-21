'
Code to run the mixed-model analysis for the mouse-task data
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'

# import packages
library(lme4)
library(lmerTest)
library(brms)
library(parameters)
library(tidyverse)
library(beanplot)
library(sjstats)
library(optimx)
library(HLMdiag)
library(DHARMa)
library(sjPlot)
library(effects)
library(grid)
library(gridExtra)
library(ggplot2)
library(broom.mixed)
library(ggeffects)
library(caret)
library(Hmisc)
library(merTools)

# to show images in a separate window
# options(device='windows')

# get the current working directory (all plots are saved in the current working directory!)
getwd()
# set the working directory to the folder with the data
setwd("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis")

##############
# Data Setup #
##############

# load the dataset
mouseTaskData <- read.csv(file = 'Mouse_Task_Features.csv')

# get some bacic stats about the dataset
# Number of rows and columns
dim(mouseTaskData)
# number of unique participants
length(table(mouseTaskData$ID))
# and their sorted number of measurements
sort(table(mouseTaskData$ID))

# list of dependent variables that will be analysed
dvs <- c("arousal", "valence")
# list of predictors (dependent variables) that will (potentially) be analysed
ivs <- c(
  'task_duration', 'clicks', 'task_total_dist', 'task_speed_mean',
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
  'trial_sd_y_flips'
)
length(ivs)


#####################################
#### Creating different datasets ####
#####################################

# there are many possible specifications to the datasets, which might produce different results in the later analysis
# In most preprocessing steps, decisions need to be made without a clear "best option"
# We therefore followed the idea of a multiverse analysis approach (Steegen et al., 2016)
# and created multiply datasets to run the data analysis with. There are potentially
# infinite possibilities for data preprocessing and our analysis is already quite large
# and complex. Our "multiverse" therefore only includes datasets with different outlier
# removal procedures. In other preprocessing steps, we made a decision for a specific option-
# Notice that every decision can be debated and should not be taken for the "truth".
# However, it was necessary to narrow down the set of possibilites in order to
# handle computations and results. The decisions are taken to the best of our knowledge
# and with regard to the input from reviewers.

# More specifically, preprocessing included multiple steps in oder to "prepare"
# the data for the mixed-model analysis

# (1) We removed outliers according to three criteria: only remove trials with a very long task duration (> 5 minutes)
# as well as using the interquantile range to remove outliers with 2 different thresholds. Note that the IQR outliers
# are removed separately for each mouse task

# (2) To account for potential systematic differences in mouse usage between
# the different mouse tasks, we used linear equating (i.e. standardize the features by the task)
# see: linear equating of the different tasks, see for example: Singh, R. K. (2021). The new normal: Linear equating of
# different instruments. DOI: 10.34879/gesisblog.2021.33)
# the idea is to threat the different tasks as "different measurement insturements" of the same underlying (behavioral)
# construct and equate the different mouse usage parameters to have the same mean and standard deviation (standardize
# the parameters by the mouse task)

# (3) To reduce redundancies between predictors in the dataset, highly correlated
# predictors will be removed using a correlation threshold of .80. For each
# correlated predictor pair, the predictor with the higher correlation with the
# other predictors is removed

# (4) Predictor and target values barely follow a normal-distribution shape (see the descriptive stats)
# To acommodate the suggestion of one reviewer that "linear Mixed Effect Model analysis
# is susceptible to the distribution of the independent and dependent variables, we transformed
# the input and output variable using rank-based inverse normal transformation

# (5) The predictor variable in the longitudinal measurement includes both,
# within-variance (state-variance) and between-variance (trait-variance). To better
# understand the relationship between the target and predictor on the trait and 
# state level, it is suggested to split the predictor into two predictors, which
# will separately account for the state (within) and trait (between)
# (e.g. see Hoffman, L. (2015). Longitudinal analysis: Modeling within-person fluctuation and change)
# Therefore, each predictor in each dataset is split in such a way

# Differenting between 3 outlier removal procedures results in three different
# datasets that will be analysed. Note, that there are many more options that could potentially be tested 
# (e.g. using no/other transformation procedures).

### helper functions for the dataset preprocessing procedure ###

# custom outlier detection function for filtering in the dyplr pipeline based on the interquartile range
# the interquartile range was choosen because it is robust against outliers in the data as compared to outlier removal
# procedures based on the z_values
iqr_outlier <- function(variable, thresh = 3.0, name) {
  # calculate the quantiles and the interquartile range
  q <- quantile(variable, c(0.25, 0.75))
  iqr <- diff(q)
  # select the iqr outliers based on a given threshold
  iqr_out <- (variable < q[1] - iqr*thresh) | (variable > q[2] + iqr*thresh)
  # print(paste0("Number of outliers for variable ", name, " = ", sum(iqr_out)))
  iqr_out
}

# remove outliers from the dataset based on the chosen method
# this is a naive implementation that can be customized if other outlier removal methods are tested out
# Note that for the IQR-outlier removal procedure, the outliers are removed by task
remove_outliers <- function (data, method, thresh = 3.0) {
  
  thresh <- as.double(thresh)

  # get the number of datapoints
  num_datapoints <- nrow(data)

  # remove outliers based on the chosen method
  if (method == "cutoff") {
    # the cutoff method just removed cases in which the task duration exceeds a selected task duration threshold
    # of 5 minutes, which removes 2 participants who had extremly long task times (basically include everyone
    # in the analysis)
   data <- data %>% filter(task_duration < 300)

  } else if (method == "iqr_out") {

    # probably not a clean TidyR approach, but working. Applies the iqr outlier function to all columns and mark
    # all rows, which include at least one outlier in any column, then remove those from the dataset (plus the
    # created outlier column)
    data$outlier <- rowSums(sapply(colnames(data[ivs]), function (x) iqr_outlier(data[[x]], thresh = thresh, x)))
    data <- data %>% filter(outlier == 0) %>% dplyr::select(-outlier)

  } else {
    stop("Chosen method is not specified")
  }
  # if an outlier removal procedure was chosen
  # print the number of removed participants using the specified filter technique
  print(paste("Number of removed participants:", num_datapoints - nrow(data)))
  # return the filtered dataframe
  return (data)

}

# function to standardize across selected variables
standardize_across_variables <- function (data, vars) {
  
  data <- data %>%
    # mutate the selected dv columns and standardize them by the entire sample
    mutate(across(.cols = all_of(vars),
                  ~ (.x - mean(.x)) / sd(.x))) %>%
    # note that the by participant standardization produces NaNs for the locksreen variables because for some
    # participants, the standard deviation is 0. Therefore, replace the NaNs with 0s
    mutate(across(.cols = all_of(vars),
                  ~ replace(., is.nan(.), 0)))
  # return the mutated dataframe
  return(data)
}

# function to perform the rank-based inverse normal transformation for the
# mouse usage features and target variable following the formular of
# Bishara, A. J., & Hittner, J. B. (2012). Testing the significance of a correlation
# with nonnormal data: comparison of Pearson, Spearman, transformation, and 
# resampling approaches. Psychological Methods, 17(3), 399-417.
# https://doi.org/10.1037/a0028087
rank_based_inverse_normal_transform <- function(x)
{
  qnorm((rank(x, na.last = "keep") - 0.5) / sum(!is.na(x)))
}

# initialize the dataset list that will iterated in the data analysis step
dataset_list <- list()

# list all the outlier removal procedures
outlier_options <- list(a=c(method="cutoff", thresh=0), b=c(method="iqr_out", thresh=2.5), c=c(method="iqr_out", thresh=3.5))

# loop the options by row, create a dataset for each option combination and add the dataset to the datasets list
for (option in outlier_options) {

  # get the raw data
  data_to_transform <- mouseTaskData

  # get the outlier method and threshold (for IQR)
  out_method <- option[["method"]]
  out_thresh <- option[["thresh"]]
  
  print(paste0("Creating a Dataset with outlier method: ", out_method,
               " and threshold", out_thresh))

  # data transformation for all datasets

  # Step 1: Remove outliers by task & standarize the ivs by the task to linear equate potential task differences
  data_to_transform <- mouseTaskData %>%
    # first, remove the outliers per task (grouped)
    group_by(taskNum) %>%
    group_modify(~ remove_outliers(data = ., method = out_method, thresh = out_thresh)) %>%
    # finally, ungroup
    ungroup()
  
  # Step 2: Linear Equate all Ivs by the mouse task
  data_to_transform <- data_to_transform %>%
    group_by(taskNum) %>%
    # second, standardize all ivs by the task (linear equate)
    standardize_across_variables(data = ., vars = ivs) %>%
    ungroup()

  # Step 3: remove highly correlated features from the dataset
  # create a correlation table of all ivs
  corr_table <- data_to_transform %>% dplyr::select(all_of(ivs)) %>% cor(.)
  # find the index of all bivariate correlations above the specified threshold using the caret::findCorrelation func
  # https://www.rdocumentation.org/packages/caret/versions/6.0-90/topics/findCorrelation
  index <- caret::findCorrelation(corr_table, .80, exact = FALSE)
  # get the names of the columns that will be removed
  to_be_removed <- colnames(corr_table)[index]
  # remove the columns from the dataframe
  data_to_transform <- data_to_transform %>%  dplyr::select(-to_be_removed)
  
  # get the remaining predictors
  remaining_predictors <- ivs[ivs %in% colnames(data_to_transform)]
  
  # Step 4: Transform the IV and DV using rank-based inverse normal transformation
  data_to_transform <- data_to_transform %>%
    mutate(across(.cols = all_of(c(remaining_predictors, dvs)), rank_based_inverse_normal_transform))
  
  # Step 5: For every predictor, separate the predictor into its state and trait
  # component -> the trait is the person mean of the predictor and the state
  # is the person mean centered predictor variable
  data_to_transform <- data_to_transform %>% 
    add_column(demean(., select = c(remaining_predictors), group = "ID"))

  # Step 6: add the transformed dataset to the list of all datasets
  dataset_list[[paste0(out_method, "_", out_thresh)]] <- data_to_transform

}


# -- Do a test visualization of the predictors and outcomes in one of the df --#

options(device = "windows")

dataset_list[[1]] %>% dplyr::select(all_of(ivs[ivs %in% colnames(dataset_list[[1]])])) %>% hist.data.frame(.)
dataset_list[[1]] %>% dplyr::select(arousal, valence) %>% hist.data.frame(.)


###############################################################################################
############################# Mixed Models Analysis Code ######################################
###############################################################################################

#################################
### Plotting Helper Functions ###
#################################

# function to create and save mixed model diagnostic plots
model_diagnostic_plots <- function (model, filename, data) {

  # create a PDF that has both diagnostic visualization plots on a seperate page
  # the PDFs are saved in the current working directory
  pdf(paste0('Task_',filename,".pdf"),
      width = 10, height = 8,
      bg = "white",
      colormodel = "cmyk",
      paper = "a4r")

    # model diagnostics visualization
    diagnostic_plots.1 <- DHARMa::simulateResiduals(fittedModel = model, plot = F)
    diagnostic_plots.2 <- sjPlot::plot_model(model, type='diag')

    # save the first plot
    plot(diagnostic_plots.1)
    # save the second plot
    grid.arrange(diagnostic_plots.2[[1]], diagnostic_plots.2[[2]]$ID, diagnostic_plots.2[[3]], diagnostic_plots.2[[4]], nrow = 2)

    dev.off()

}


# function to create and save plots about the relationship between a single predictor variable and the target
# and taking the multilevel structure into account
# all Plots are marginal effect plot of the predictor variable on the target variable controlling for the covariates
plot_single_pred_mixed_model <- function (dataset, predictor, target) {

  print("Plotting the Invididual Linear Models")

  # First, plot the relationship between predictor and target when ignoring the group structure plus individual
  # relationship plots per participant (individual regression models)

  # calculate the marginal effects between target and predictor per participant
  marg_effect_predictions <- dataset%>%
    group_by(ID) %>%
    do(preds = ggpredict(glm(data = ., formula = as.formula(paste0(target, " ~ ", predictor)),
                             family = "gaussian"), terms = predictor))

  # create a single dataset for plotting from the predicted marginal effects per group and add a group label
  merged_marg_eff_preds <- bind_rows(marg_effect_predictions$preds, .id = "par")

  # plot the invididual regression lines
  lm_plot <- ggplot(data = merged_marg_eff_preds, aes(x = x, y = predicted, colour = par)) +
    # add participant regression line
    geom_line(size = 1, alpha = 0.5) +
    # add the regression line of the total dataset when ignoring the grouped data structure
    geom_line(data = ggpredict(glm(formula = as.formula(paste(target, " ~ ", predictor)),
                                   data = dataset, family = "gaussian"), terms = predictor),
              aes(x = x, y = predicted), colour = "black", size = 2) +
    # add the conf intervals of the regression line
    geom_ribbon(data = ggpredict(glm(formula = as.formula(paste(target, " ~ ", predictor)),
                                     data = dataset, family = "gaussian"), terms = predictor),
                aes(x = x, y = predicted, ymin = conf.low, ymax = conf.high, colour = NULL), alpha =0.2) +
    theme_light() +
    xlab(predictor) + ylab(target) +
    labs(title = paste0(target, " by ", predictor),
         subtitle = "Model per Group and Overall") +
    theme(legend.position="none")

  # Second, Plot the random intercept model with individual slopes per participant + the general slope of the random
  # intercept model
  
  print("Plotting the Fixed Effect Mixed Model")
  
  # calculate the fixed effect model first
  fe_formular <- paste(target, '~', paste0(predictor, "_within"), '+',paste0(predictor, "_between"), '+ (1|ID)')
  fixed_effect_model <- lmer(formula = as.formula(fe_formular), data = dataset, REML = F, control=lmerControl(optimizer = "bobyqa"))
  
  # now plot it
  ri_plot <- ggplot(data = ggpredict(fixed_effect_model, terms = c(paste0(predictor, "_within"))), aes(x = x, y = predicted)) +
    # linear model for each participant with unique intercept
    geom_line(data = ggpredict(fixed_effect_model, terms = c(paste0(predictor, "_within"), "ID"), type = "re"),
              aes(x = x, y = predicted, colour = group),
              size = 1, alpha = 0.5) +
    # general trend of the random intercept model
    geom_line(colour = "black", size = 2) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha =0.2) +
    theme_light() +
    xlab(predictor) + ylab(target) +
    labs(title = paste0(target, " by ", predictor),
         subtitle = "Random Intercept Model") +
    theme(legend.position="none")

  # Third, Plot the random intercept and random slopes model + the general slope of the random intercept + slope model
  print("Plotting the Random Slope Mixed Model")
  
  # calculate the random slope model first
  re_formular <- paste(target, '~', paste0(predictor, "_within"), '+',paste0(predictor, "_between"), '+ (', paste0(predictor, "_within"), '|ID)')
  random_effect_model <- lmer(formula = as.formula(re_formular), data = dataset, REML = F, control=lmerControl(optimizer = "bobyqa"))
  
  # now plot it
  rs_plot <- ggplot(data = ggpredict(random_effect_model, terms = c(paste0(predictor, "_within"))), aes(x = x, y = predicted)) +
    # linear relationships for each participant
    geom_line(data = ggpredict(random_effect_model, terms = c(paste0(predictor, "_within"), "ID"), type = "re"),
              aes(x = x, y = predicted, colour = group),
              size = 1, alpha = 0.5) +
     # general trend of the random slope model
    geom_line(color = "black", size = 2) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
    theme_light() +
    xlab(predictor) + ylab(target) +
    labs(title = paste0(target, " by ", predictor),
         subtitle = "Random Intercept and Slope Model") +
    theme(legend.position="none")


  # save the plots in a PDF file (each plot on an individual page in the file)
  # different pathname depending on the model family (binomial vs. gaussian)
  save_path <- paste0("Task_single_pred_", predictor, '_', target, ".pdf")
  # create the pdf in which the plots are saved
  pdf(save_path,
      width = 10, height = 8,
      bg = "white",
      colormodel = "cmyk",
      paper = "a4r")

  # save the lm plot
  plot(lm_plot)
  # save the random intercept plot on a seperate page
  plot(ri_plot)
  # save the random intercept + slope plot on a seperate page
  plot(rs_plot)

  dev.off()
}


#############################################################
### Helper Function to fit different mixed effect models ###
#############################################################


# fit the specified mixed model, get model diagnostics (if wanted) and return
# model coefficients as well as model performance criteria
fit_mixed_model <- function(dataset, model_formular, mod_name, plot_diag = F) {
  
  # cat(sprintf("Fitting the Model: %s\n", mod_name))
  
  # fit the model with the specified formular
  mod <- lmer(formula = as.formula(model_formular), data = dataset, REML = F, control=lmerControl(optimizer="bobyqa"))
  
  # if specified, get model diagnostic visualizations
  if (plot_diag) {model_diagnostic_plots(model = mod, filename = mod_name, data = dataset) }
  
  # grab the model coefficients
  coefficients <- broom.mixed::tidy(mod, conf.int = T, conf.method = "Wald")
  
  # calculate standardized model coefficients using the parameters package
  # which implements a method from Hoffman, 2015
  # get standardized model parameters
  standardized_coeffs <- parameters::standardise_parameters(mod, method = "pseudo", ci_method = "Wald")
  
  # calculate the model performance criteria
  model_diag <- performance::model_performance(mod)
  
  # calculate another R² value as the squared correlation between the
  # response variabe the and predicted values, which is suggested
  # by Hoffman et al., 2015 and also mentioned in a blog post about mixed model
  # goodness-of-fit coefficients by the author of the lme4 package
  # http://bbolker.github.io/mixedmodels-misc/glmmFAQ.html#how-do-i-compute-a-coefficient-of-determination-r2-or-an-analogue-for-glmms
  explained_var <- cor(model.response(model.frame(mod)),predict(mod,type="response"))^2
  # add ot to the model diagnostic criteria
  model_diag["pseudo_R2"] <- explained_var
  
  # return the model, the model coefficients and the performance criteria in a list
  list(mod = mod, coeffs = coefficients, std_coeffs=standardized_coeffs,
       model_diag = model_diag)
  
}


##############################################################
### Test the functions for selected datasets and variables ###
##############################################################
# Before Looping all datasets and variables, the mixed model results can be extracted for specified datasets and variables
# This is a playground to check to get single results, plots etc...

# the mixed model functions require a dataset, a target and a predictor (or 2 for the interaction effect) and the model
# family
# Here, random inputs are selected, but this can be adapted to manually selected options
# select a random item from the list, but leave the list inplace in order to get the name of the dataset
# write a small helper function (get random items or put them in manually)
get_sample_data <- function (dset = NULL, target = NULL, predictor = NULL) {

  # draw a random dataset if no dataset is specified
  play_dataset <- if (!is.null(dset)) dset else dataset_list[sample(seq_along(dataset_list), 1)]
  # draw a random target variable
  play_target <- if (!is.null(target)) target else sample(dvs, 1)
  # draw a random iv from the sample dataset
  play_pred <- if (!is.null(predictor)) target else sample(ivs[ivs %in% colnames(play_dataset[[1]])], 1)

  list("dset" = play_dataset, "target" = play_target, "pred" = play_pred)

}

# get a random dataset, the predictor and the target
rng_play_dat <- get_sample_data()
rng_pred <- rng_play_dat[["pred"]]
rng_target <- rng_play_dat[["target"]]

# Try out the model functions with the randomly generated data

# icc model results (create and save a model fit plot)
play_icc_mod_results <-  fit_mixed_model(dataset = rng_play_dat[["dset"]][[1]],
                                         model_formular = paste(rng_target, '~', '1 + (1|ID)'),
                                         mod_name = paste0("NullMod_", rng_target, "_" ,names(rng_play_dat["dset"][[1]])),
                                         plot_diag = T)

# single predictor model results (with plots)
play_sing_pred_fixed <- fit_mixed_model(dataset = rng_play_dat[["dset"]][[1]],
                                      model_formular = paste(rng_target, '~', paste0(rng_pred, "_within"), "+", paste0(rng_pred, "_between"), '+ (1|ID)'),
                                      mod_name = paste0("FixedEffMod_", rng_target, "_", rng_pred, "_", names(rng_play_dat["dset"][[1]])),
                                      plot_diag = T)

# single predictor model results (with plots)
play_sing_pred_slope <- fit_mixed_model(dataset = rng_play_dat[["dset"]][[1]],
                                      model_formular = re_form <- paste(rng_target, '~', paste0(rng_pred, "_within"), "+", paste0(rng_pred, "_between"), '+ (', paste0(rng_pred, "_within"), '|ID)'),
                                      mod_name = paste0("RandSlopeMod_", rng_target, "_", rng_pred, "_", names(rng_play_dat["dset"][[1]])),
                                      plot_diag = T)

# create a plot of the effect of the predictor on the outcome variable
plot_single_pred_mixed_model(dataset = rng_play_dat[["dset"]][[1]],
                             predictor = rng_play_dat[["pred"]],
                             target = rng_play_dat[["target"]])


##############################
### Model Calculation Loop ###
##############################

# TODO: Also get the ICC values of the mouse usage features as descriptive stats?

# This section includes the loop to calculate all specified models using the data analysis helper functions
# Note that each loop involves multiple model calculations and can take some time.
# See the "Playground" section for options to calculate single models for selected dataframes, model specifications
# and variables

# simple helper function to save the model results as separate csv files
save_model_results <- function (model_list) {
  # loop the model results list and save the results
  lapply(names(model_list), function (x) write.csv(x=model_list[[x]], file = paste0(x, ".csv"), row.names = F))
}

# setup lists to store the model result of the loops
ri_coeff_list <- list()
ri_diag_list <- list()

fe_coeff_list <- list()
fe_std_coeff_list <- list()
fe_diag_list <- list()

rs_coeff_list <- list()
rs_std_coeff_list <- list()
rs_diag_list <- list()

model_comparison <- list()

mouse_icc <- list()


# loop over all dataframes
for (i in seq_along(dataset_list)) {
  dset_name <- names(dataset_list[i])
  dset <- dataset_list[[i]]
  # loop over all dependent variables
  for (dv in dvs) {
    print(paste("Calculating the ICC Model for dataset", dset_name, "and target", dv))
    
    # first calculate the random intercept model for the outcome variable
    icc_model <- fit_mixed_model(dataset = dset,
                                 model_formular = paste(dv, '~', '1 + (1|ID)'),
                                 mod_name = paste0("NullMod_", dv, "_" , dset_name),
                                 plot_diag = F)
    # extract the results from the model, add the info about the dv and dataset
    icc_coeffs <- icc_model[["coeffs"]] %>% mutate(dv = dv, dframe = dset_name)
    icc_diag <- icc_model[["model_diag"]] %>% mutate(dv = dv, dframe = dset_name)
    # save them in the list
    ri_coeff_list[[paste0(dset_name, "_", dv)]] <- icc_coeffs
    ri_diag_list[[paste0(dset_name, "_", dv)]] <- icc_diag
    
    # next, loop all predictors and calculate the predictor models
    # to do so, first get the name of the remaining predictors in the dataset
    # after preprocessing
    remaining_ivs <- ivs[ivs %in% colnames(dset)]
    # next, loop the remaining ivs
    for (iv in remaining_ivs) {
      print(paste("Calculating the models predictor", iv, " in dataset", dset_name, "and target", dv))
      # calculate the fixed effect model first
      fe_model <- fit_mixed_model(dataset = dset,
                                  model_formular = paste(dv, '~', paste0(iv, "_within"), "+", paste0(iv, "_between"), '+ (1|ID)'),
                                  mod_name = paste0("FE_Mod", iv, "_", dv, "_" , dset_name),
                                  plot_diag = F)
      # calculate the random effect model second
      rs_model <- fit_mixed_model(dataset = dset,
                                  model_formular = paste(dv, '~', paste0(iv, "_within"), "+", paste0(iv, "_between"), '+ (', paste0(iv, "_within"), '|ID)'),
                                  mod_name = paste0("RE_Mod", iv, "_", dv, "_" , dset_name),
                                  plot_diag = F)
      
      # Again, extract the infos from the models and save them in the lists
      fe_coeffs <- fe_model[["coeffs"]] %>% mutate(iv = iv, dv = dv, dframe = dset_name)
      fe_std_coeffs <- fe_model[["std_coeffs"]] %>% mutate(iv = iv, dv = dv, dframe = dset_name)
      fe_diag <- fe_model[["model_diag"]] %>% mutate(iv = iv, dv = dv, dframe = dset_name)
      fe_coeff_list[[paste0(iv, "_", dset_name, "_", dv)]] <- fe_coeffs
      fe_std_coeff_list[[paste0(iv, "_", dset_name, "_", dv)]] <- fe_std_coeffs
      fe_diag_list[[paste0(iv, "_", dset_name, "_", dv)]] <- fe_diag
      
      rs_coeffs <- rs_model[["coeffs"]] %>% mutate(iv = iv, dv = dv, dframe = dset_name)
      rs_std_coeffs <- rs_model[["std_coeffs"]] %>% mutate(iv = iv, dv = dv, dframe = dset_name)
      rs_diag <- rs_model[["model_diag"]] %>% mutate(iv = iv, dv = dv, dframe = dset_name)
      rs_coeff_list[[paste0(iv, "_", dset_name, "_", dv)]] <- rs_coeffs
      rs_std_coeff_list[[paste0(iv, "_", dset_name, "_", dv)]] <- rs_std_coeffs
      rs_diag_list[[paste0(iv, "_", dset_name, "_", dv)]] <- rs_diag
      
      # then, compare the nested models
      comparison <- performance::test_performance(icc_model[["mod"]],
                                                  fe_model[["mod"]],
                                                  rs_model[["mod"]]) %>%
        mutate(iv = iv, dv = dv, dframe = dset_name)
      # and add the it to the results list
      model_comparison[[paste0(iv, "_", dset_name, "_", dv)]] <- comparison
      
      # finally, also calculate the ICC for a random intercept only model with
      # the mouse usage predictor as the dependent variable to check how much
      # between-person variance and within-person variance the mouse usage
      # features have
      mouseICC <- merTools::ICC(outcome = iv, group = "ID", data = dset)
      icc_df <- data.frame (ICC  = c(mouseICC),
                        iv = c(iv),
                        dframe = c(dset_name))
      # add it to the results list
      mouse_icc[[paste0(iv, "_", dset_name, "_", dv)]] <- icc_df
      
    }
  }
}

# convert each list to a "final results" dataframe and save the final results in a list
mouse_task_results <- list("Task_results_ri_coeffs" = dplyr::bind_rows(ri_coeff_list),
                   "Task_results_ri_diag" = dplyr::bind_rows(ri_diag_list),
                   "Task_results_fe_coeffs" = dplyr::bind_rows(fe_coeff_list),
                   "Task_results_fe_std_coeffs" = dplyr::bind_rows(fe_std_coeff_list),
                   "Task_results_fe_diag" = dplyr::bind_rows(fe_diag_list),
                   "Task_results_rs_coeffs" = dplyr::bind_rows(rs_coeff_list),
                   "Task_results_rs_std_coeffs" = dplyr::bind_rows(rs_std_coeff_list),
                   "Task_results_rs_diag" = dplyr::bind_rows(rs_diag_list),
                   "Task_results_model_comparison" = dplyr::bind_rows(model_comparison),
                   "Task_results_MousePred_ICC" = dplyr::bind_rows(mouse_icc))

# save the results as csv files
save_model_results(mouse_task_results)

# If already Calculated: Import the results from the CSV files instead of running the loop
mouse_task_results <- list(
  "Task_results_ri_coeffs" = read.csv("Results_NEW/task_results_ri_coeffs.csv"),
  "Task_results_ri_diag" = read.csv("Results_NEW/task_results_ri_diag.csv"),
  "Task_results_fe_coeffs" = read.csv("Results_NEW/task_results_fe_coeffs.csv"),
  "Task_results_fe_std_coeffs" = read.csv("Results_NEW/task_results_fe_std_coeffs.csv"),
  "Task_results_fe_diag" = read.csv("Results_NEW/task_results_fe_diag.csv"),
  "Task_results_rs_coeffs" = read.csv("Results_NEW/task_results_rs_coeffs.csv"),
  "Task_results_rs_std_coeffs" = read.csv("Results_NEW/task_results_rs_std_coeffs.csv"),
  "Task_results_rs_diag" = read.csv("Results_NEW/task_results_rs_diag.csv"),
  "Task_results_model_comparison" = read.csv("Results_NEW/task_results_model_comparison.csv"),
  "Task_results_MousePred_ICC" = read.csv("Results_NEW/Task_results_MousePred_ICC")
)


#################
### Test Loop ###
#################

# run a test loop in order to compare the results of the within-between separated
# predictors and the original predictor (only use one dataset and one outcome here)

test_coeff_list <- list()
test_coeff_std_list <- list()

# loop over all dataframes
for (i in seq_along(dataset_list)) {
  dset_name <- names(dataset_list[i])
  dset <- dataset_list[[i]]
  # loop over all dependent variables
  for (dv in dvs) {
    # get remaining ivs
    remaining_ivs <- ivs[ivs %in% colnames(dset)]
    # next, loop the remaining ivs
    for (iv in remaining_ivs) {
      print(paste("Calculating the models predictor", iv, " in dataset", dset_name, "and target", dv))
      # calculate the fixed effect model first
      test_model <- fit_mixed_model(dataset = dset,
                                  model_formular = paste(dv, '~', iv, '+ (1|ID)'),
                                  mod_name = paste0("FE_Mod", iv, "_", dv, "_" , dset_name),
                                  plot_diag = F)
      
      # Again, extract the infos from the models and save them in the lists
      test_coeffs <- test_model[["coeffs"]] %>% mutate(iv = iv, dv = dv, dframe = dset_name)
      test_coeffs_std <- test_model[["std_coeffs"]] %>% mutate(iv = iv, dv = dv, dframe = dset_name)
      test_coeff_list[[paste0(iv, "_", dset_name, "_", dv)]] <- test_coeffs
      test_coeff_std_list[[paste0(iv, "_", dset_name, "_", dv)]] <- test_coeffs_std
      
    }
    break
  }
  break
}

# convert each list to a "final results" dataframe and save the final results in a list
test_results <- dplyr::bind_rows(test_coeff_list)
test_results_std <- dplyr::bind_rows(test_coeff_std_list)

# filter the relevant rows, and get the results that will be compared with the
# test results and use the same filtering aswell

test_results <- test_results %>% 
  dplyr::filter(., grepl("fixed", effect)) %>%
  dplyr::filter(., !grepl("Intercept", term)) %>%
  dplyr::select(estimate, p.value, term)
  
compare_with_test <- mouse_task_results[["Task_results_fe_coeffs"]] %>%
  dplyr::filter(., grepl("fixed", effect)) %>%
  dplyr::filter(., !grepl("Intercept", term)) %>%
  dplyr::filter(., grepl("arousal", dv)) %>%
  dplyr::filter(., grepl("cutoff_0", dframe)) %>%
  dplyr::select(estimate, p.value, term)
                  
write.csv(test_results, "Orig_Pred.csv", row.names=FALSE)
write.csv(compare_with_test, "Within_Between_Pred.csv", row.names=FALSE)

# do the same thing for the standardized coefficients

test_std_results <- test_results_std %>% 
  dplyr::filter(., !grepl("Intercept", Parameter)) %>%
  dplyr::select(Std_Coefficient, Parameter)


compare_with_test_std <- mouse_task_results[["Task_results_fe_std_coeffs"]] %>%
  dplyr::filter(., grepl("within|between", Parameter)) %>%
  dplyr::filter(., grepl("arousal", dv)) %>%
  dplyr::filter(., grepl("cutoff_0", dframe)) %>%
  dplyr::select(Std_Coefficient, Parameter)


write.csv(test_std_results, "Orig_STD_Pred.csv", row.names=FALSE)
write.csv(compare_with_test_std, "Within_Between_STD_Pred.csv", row.names=FALSE)


#############################
### Visualize the Results ###
#############################


# Implement at the top to give the dataset nicer variable names?

test <- single_pred_results[["Task_results_sp_re_coeffs"]] %>%
  # filter the relevant dv data
  filter(dv == 'arousal') %>%
  # filter out all effects that are not plotted
  filter(.data = ., !grepl('(Intercept)|sd_(Intercept)|sd__Observation', term)) %>%
  # remove the sd__ string from the random effect coefficient names
  mutate(term = str_replace(term, 'sd__', '')) %>%
  # rename the term values to better variable name values (tideous work which probably should have done in an earlier step
  mutate(term = recode(term, "clicks" = 'Clicks',
                       "task_total_dist" = "Task: Tot. Distance",
                       "task_angle_sd" = 'Task: Angle (sd)',
                       "task_x_flips" = 'Task: X-Flips',
                       "task_y_flips" = 'Task: Y-Flips',
                       "trial_sd_duration" = 'Trial (sd): Duration',
                       "trial_mean_trial_move_offset" = 'Trial (mean): Initiation Time',
                       "trial_sd_trial_move_offset" = 'Trial (sd): Initiation Time',
                       "trial_sd_total_dist" = 'Trial (sd): Tot. Distance',
                       "trial_sd_distance_overshoot" = 'Trial (sd): Ideal Line Deviation',
                       "trial_mean_speed_mean" = 'Trial (mean): Speed (mean)',
                       "trial_sd_speed_mean" = 'Trial (sd): Speed (mean)',
                       "trial_mean_speed_sd" = 'Trial (mean): Speed (sd)',
                       "trial_sd_speed_sd" = 'Trial (sd): Speed (sd)',
                       "trial_sd_abs_jerk_sd" = 'Trial (sd): Jerk (sd)',
                       "trial_mean_angle_mean" = 'Trial (mean): Angle (mean)',
                       "trial_sd_angle_mean" = 'Trial (sd): Angle (mean)',
                       "trial_mean_angle_sd" = 'Trial (mean): Angle (sd)',
                       "trial_sd_angle_sd" = 'Trial (sd): Angle (sd)',
                       "trial_sd_x_flips" = 'Trial (sd): X-Flips',
                       "trial_sd_y_flips" = 'Trial (sd): Y-Flips',
                       "trial_mean_duration" = 'Trial (mean): Duration',
                       "trial_mean_total_dist" = 'Trial (mean): Tot. Distance',
                       "trial_sd_abs_jerk_mean" = 'Trial (sd): Jerk (mean)',
                       "task_duration" = 'Task: Duration',
                       "task_abs_jerk_sd" = 'Task: Jerk (sd)',
                       "trial_mean_abs_jerk_mean" = 'Trial (mean): Jerk (mean)')) %>%
  group_split(effect)


# TODO: HOW TO CREATE THE PLOTS WITH DOUBLE THE VARIABLES? (ONLY WITHIN?)

# A side-by-side plot of the fixed effect coefficients with their confidence intervals per predictor per dataset
# for the random intercept model and the random intercept + random slope model plus the standard deviation of the
# random effects for the random slope + random intercept model
plot_coefficient_estimates <- function (fe_coeff_data, re_coeff_data, title, dot_size=1) {

  # split the random intercept + slope data into the fixed effect estimates and the random effect estimates
  # the first dataset of the list contains the fixed effect estimates, the second, the random effect estimates
  split_re_coeffs <- re_coeff_data %>% group_split(effect)

  # random intercept only fixed effect coefficient plot
  random_intercept_fixed_eff_plot <- ggplot(fe_coeff_data, aes(x=estimate, y=term, color=dframe, group=dframe)) +
    # plot a vline at 0
    geom_vline(xintercept = 0, colour = "black", linetype = 2, size=1) +
    # plot the fixed effect coefficients with their CIs
    geom_pointrange(aes(xmin = conf.low, xmax = conf.high), position=position_dodge(width = 0.6), size=dot_size) +
    # plot separator lines between the predictors
    geom_hline(yintercept = seq_along(unique(fe_coeff_data$term)) +0.5, colour = "grey60", linetype = "twodash") +
    # customize the x- and y-label
    xlab("Rand. Intercept Model: Fixed Effect Coeffs with 95% CI") +
    ylab("Predictors") +
    theme_minimal() +
    theme(text = element_text(size = 14)) +
    # disable the legend
    theme(legend.position="none")

  # random intercept plus random slope fixed effect coefficient plot
  random_intercept_slope_fixed_eff_plot <- ggplot(split_re_coeffs[[1]], aes(x=estimate, y=term, color=dframe, group=dframe)) +
    # plot a vline at 0
    geom_vline(xintercept = 0, colour = "black", linetype = 2, size=1) +
    # plot the fixed effect coefficients with their CIs
    geom_pointrange(aes(xmin = conf.low, xmax = conf.high), position=position_dodge(width = 0.6), size=dot_size) +
    # plot separator lines between the predictors
    geom_hline(yintercept = seq_along(unique(split_re_coeffs[[1]]$term)) +0.5, colour = "grey60", linetype = "twodash") +
    # customize the x- and y-label
    xlab("Rand. Intercept & Slope Model: Fixed Effect Coeffs with 95% CI") +
    ylab("") +
    theme_minimal() +
    # hide the y-axis
    theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), text = element_text(size = 14)) +
    # disable the legend
    theme(legend.position="none")

  # random effects of the slopes plot of the random intercept + slope model
  random_intercept_slope_random_plot <- ggplot(split_re_coeffs[[2]], aes(x=estimate, y=term, color=dframe, group=dframe)) +
    # plot a vline at 0
    geom_vline(xintercept = 0, colour = "black", linetype = 2, size=1) +
    # plot the random effect coefficients
    geom_point(position=position_dodge(width = 0.6), size=dot_size*3) +
    # plot separator lines between the predictors
    geom_hline(yintercept = seq_along(unique(split_re_coeffs[[2]]$term)) +0.5, colour = "grey60", linetype = "twodash") +
    # customize the labels and the legend text
    labs(x="Random Effect Standard Deviations", y="", color="Datasets") +
    scale_color_hue(labels = c("dur<5min & z-Par", "2.5*IQR & z-Par", "3.5*IQR & z-Par", "dur<5min & z-Samp", "2.5*IQR & z-Samp", "3.5*IQR & z-Samp")) +
    theme_minimal() +
    # hide the y-axis
    theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), text = element_text(size = 14), legend.text = element_text(size = 14))

  # merge the plots together
  merged_plot <- arrangeGrob(random_intercept_fixed_eff_plot,
                             random_intercept_slope_fixed_eff_plot,
                             random_intercept_slope_random_plot,
                             ncol = 3, widths = c(1, 0.75, 1),
                             top = textGrob(str_replace_all(title, "_", " "))
  )

  # save the plot
  # use different figure sizes depending on what plot needs to be saved
  # for the single predictor result plots, use a width of 20 and height of 12
  # for the interaction result plots, use a width of 24 and a height of 14
  ggsave(paste0(title,".png"), merged_plot, width = 20, height = 12)

  dev.off()

}


# Save a plot for every dependent variable
for (target in dvs) {
  # get the (cleaned) coefficient data for the random intercept only model
  fe_coeffs <- single_pred_results[["Task_results_sp_fe_coeffs"]] %>%
    # filter the relevant dv data
    filter(dv == target) %>%
    # filter out all effects that are not plotted (only the fixed effects are plotted
    filter(.data = ., !grepl('(Intercept)|timestamp|zoom|screen_width|screen_height|median_sampling_freq|sd_(Intercept)|sd__Observation', term)) %>%
    # rename the term values to better variable name values (tideous work which probably should have done in an earlier step
    mutate(term = recode(term, "clicks" = 'Clicks',
                         "task_total_dist" = "Task: Tot. Distance",
                         "task_angle_sd" = 'Task: Angle (sd)',
                         "task_x_flips" = 'Task: X-Flips',
                         "task_y_flips" = 'Task: Y-Flips',
                         "trial_sd_duration" = 'Trial (sd): Duration',
                         "trial_mean_trial_move_offset" = 'Trial (mean): Initiation Time',
                         "trial_sd_trial_move_offset" = 'Trial (sd): Initiation Time',
                         "trial_sd_total_dist" = 'Trial (sd): Tot. Distance',
                         "trial_sd_distance_overshoot" = 'Trial (sd): Ideal Line Deviation',
                         "trial_mean_speed_mean" = 'Trial (mean): Speed (mean)',
                         "trial_sd_speed_mean" = 'Trial (sd): Speed (mean)',
                         "trial_mean_speed_sd" = 'Trial (mean): Speed (sd)',
                         "trial_sd_speed_sd" = 'Trial (sd): Speed (sd)',
                         "trial_sd_abs_jerk_sd" = 'Trial (sd): Jerk (sd)',
                         "trial_mean_angle_mean" = 'Trial (mean): Angle (mean)',
                         "trial_sd_angle_mean" = 'Trial (sd): Angle (mean)',
                         "trial_mean_angle_sd" = 'Trial (mean): Angle (sd)',
                         "trial_sd_angle_sd" = 'Trial (sd): Angle (sd)',
                         "trial_sd_x_flips" = 'Trial (sd): X-Flips',
                         "trial_sd_y_flips" = 'Trial (sd): Y-Flips',
                         "trial_mean_duration" = 'Trial (mean): Duration',
                         "trial_mean_total_dist" = 'Trial (mean): Tot. Distance',
                         "trial_sd_abs_jerk_mean" = 'Trial (sd): Jerk (mean)',
                         "task_duration" = 'Task: Duration',
                         "task_abs_jerk_sd" = 'Task: Jerk (sd)',
                         "trial_mean_abs_jerk_mean" = 'Trial (mean): Jerk (mean)'))

  # get the (cleaned) coefficient data for the random intercept + slope model
  re_coeffs <- single_pred_results[["Task_results_sp_re_coeffs"]] %>%
    # filter the relevant dv data
    filter(dv == target) %>%
    # filter out all effects that are not plotted
    filter(.data = ., !grepl('(Intercept)|timestamp|zoom|screen_width|screen_height|median_sampling_freq|sd_(Intercept)|sd__Observation', term)) %>%
    # remove the sd__ string from the random effect coefficient names in order to rename it in the next step
    mutate(term = str_replace(term, 'sd__', '')) %>%
    # rename the term values to better variable name values (tideous work which probably should have done in an earlier step
    mutate(term = recode(term, "clicks" = 'Clicks',
                         "task_total_dist" = "Task: Tot. Distance",
                         "task_angle_sd" = 'Task: Angle (sd)',
                         "task_x_flips" = 'Task: X-Flips',
                         "task_y_flips" = 'Task: Y-Flips',
                         "trial_sd_duration" = 'Trial (sd): Duration',
                         "trial_mean_trial_move_offset" = 'Trial (mean): Initiation Time',
                         "trial_sd_trial_move_offset" = 'Trial (sd): Initiation Time',
                         "trial_sd_total_dist" = 'Trial (sd): Tot. Distance',
                         "trial_sd_distance_overshoot" = 'Trial (sd): Ideal Line Deviation',
                         "trial_mean_speed_mean" = 'Trial (mean): Speed (mean)',
                         "trial_sd_speed_mean" = 'Trial (sd): Speed (mean)',
                         "trial_mean_speed_sd" = 'Trial (mean): Speed (sd)',
                         "trial_sd_speed_sd" = 'Trial (sd): Speed (sd)',
                         "trial_sd_abs_jerk_sd" = 'Trial (sd): Jerk (sd)',
                         "trial_mean_angle_mean" = 'Trial (mean): Angle (mean)',
                         "trial_sd_angle_mean" = 'Trial (sd): Angle (mean)',
                         "trial_mean_angle_sd" = 'Trial (mean): Angle (sd)',
                         "trial_sd_angle_sd" = 'Trial (sd): Angle (sd)',
                         "trial_sd_x_flips" = 'Trial (sd): X-Flips',
                         "trial_sd_y_flips" = 'Trial (sd): Y-Flips',
                         "trial_mean_duration" = 'Trial (mean): Duration',
                         "trial_mean_total_dist" = 'Trial (mean): Tot. Distance',
                         "trial_sd_abs_jerk_mean" = 'Trial (sd): Jerk (mean)',
                         "task_duration" = 'Task: Duration',
                         "task_abs_jerk_sd" = 'Task: Jerk (sd)',
                         "trial_mean_abs_jerk_mean" = 'Trial (mean): Jerk (mean)'))

  # feed the datasets into the visualization function
  plot_coefficient_estimates(fe_coeffs, re_coeffs, paste0("Task_single_predictor_estimates_for_target_", target),
                             dot_size = .75)

}


############################################
### Additional Control Variable Analysis ###
############################################

# get a random dataset, the predictor and the target
control_data <- get_sample_data()
control_dset <- control_data[["dset"]][[1]]
control_pred <- control_data[["pred"]]
control_target <- control_data[["target"]]

# create an order variable for each participant from the timestamp variable
control_dset <- control_dset %>%
  group_by(ID) %>%
  arrange(., timestamp) %>%
  mutate(., order = 1:n()) %>%
  ungroup()

# list the control variables
control_variables <- c('hand', 'sex', 'age', 'order', 'zoom', 'screen_width', 
                'screen_height', 'median_sampling_freq')

# run a model only with the control variables
control_model <- fit_mixed_model(dataset = control_dset,
                                model_formular = paste(control_target, '~', paste(control_variables, collapse = "+"), '+ (1|ID)'),
                                mod_name = paste0("Control Only Model"),
                                plot_diag = F)

# run a model with the predictor and all control variables
control_pred_model <- fit_mixed_model(dataset = control_dset,
                                      model_formular = paste(control_target, '~', paste(control_variables, collapse = "+"), '+', control_pred, '+ (1|ID)'),
                                      mod_name = paste0("Control and Predictor Model"),
                                      plot_diag = F)

performance::test_performance(control_model[["mod"]], control_pred_model[["mod"]])

# run an interaction model with sex and mouse usage
sex_interaction <- fit_mixed_model(dataset = control_dset,
                                   model_formular = paste(control_target, '~ sex *', control_pred, '+ (1|ID)'),
                                   mod_name = paste0("Sex Interaction Model"),
                                   plot_diag = F)

sex_interaction[["coeffs"]]

# run a model with mouse usage as the target and order as a predictor to check
# for a time effect on mouse usage
order_effect <- fit_mixed_model(dataset = control_dset,
                                model_formular = paste(control_pred, '~ order + (1|ID)'),
                                mod_name = paste0("Order Effect Model"),
                                plot_diag = F)

order_effect[["coeffs"]]

# run a model to check for an order * mouse usage interaction
order_interaction <- fit_mixed_model(dataset = control_dset,
                                     model_formular = paste(control_target, '~ order *', control_pred, '+ (1|ID)'),
                                     mod_name = paste0("Order Interaction Model"),
                                     plot_diag = F)

order_interaction[["coeffs"]]
