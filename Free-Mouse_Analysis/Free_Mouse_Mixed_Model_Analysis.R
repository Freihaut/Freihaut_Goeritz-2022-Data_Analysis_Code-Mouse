'
Code to run the mixed-model analysis for the mouse-task data
For questions regarding the code, please contact: paul.freihaut@psychologie.uni-freiburg.de
'

# package import
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
library(bestNormalize)
library(caret)

# plot to windows (because Pycharm does not show plots in R)
options(device='windows')

# get the current working directory (all plots and data are saved in the current working directory!)
wd <- getwd()

##############
# Data Setup #
##############

# load the dataset
# Note that R automatically adds an X before the numbers in the column names,
# e.g. column name: 3000_recording_duration -> X3000_recording_duration
mouse_data <- read.csv(file = 'Free_Mouse_Features.csv')

# get some bacic stats about the dataset
# Number of rows and columns
dim(mouse_data)
# number of unique participants
length(table(mouse_data$ID))
# and their sorted number of measurements
sort(table(mouse_data$ID))

# list of dependent variables that will be analysed
dvs <- c("arousal", "valence", "stress")

# list of predictors (dependent variables) that will (potentially) be analysed
ivs <- c(
  'recording_duration', 'movement_episodes', 'mo_ep_mean_episode_duration',
  'mo_ep_sd_episode_duration', 'mo_ep_mean_total_dist', 'mo_ep_sd_total_dist',
  'mo_ep_mean_speed_mean', 'mo_ep_sd_speed_mean', 'mo_ep_mean_speed_sd', 'mo_ep_sd_speed_sd',
  'mo_ep_mean_abs_accel_mean', 'mo_ep_sd_abs_accel_mean', 'mo_ep_mean_abs_accel_sd',
  'mo_ep_sd_abs_accel_sd', 'mo_ep_mean_abs_jerk_mean', 'mo_ep_sd_abs_jerk_mean',
  'mo_ep_mean_abs_jerk_sd', 'mo_ep_sd_abs_jerk_sd', 'mo_ep_mean_angle_mean',
  'mo_ep_sd_angle_mean', 'mo_ep_mean_angle_sd', 'mo_ep_sd_angle_sd', 'mo_ep_mean_x_flips',
  'mo_ep_sd_x_flips', 'mo_ep_mean_y_flips', 'mo_ep_sd_y_flips', 'movement_duration',
  'movement_distance', 'no_movement', 'lockscreen_episodes.', "lockscreen_time"
)
length(ivs)

# list the covariates that will be included in the mixed models
covariates <- c('timestamp', 'zoom', 'screen_width', 'screen_height')

# split all potential predictors into accuracy and speed predictors, this distinction will be used to create
# speed-accuracy feature pairs in order to test (replicate) the effect of stress (emotional states) on a
# speed-accuracy tradeoff (note that in the free mouse usage dataset, specifying indicators for speed and accuracy
# is not as clear as in the mouse task. The mouse task had an ideal movement path from point to point, which allows
# to specify the accuracy of the movement and the speed of the movement. In the free mouse recording setting, the
# movement goal is unclear and there is no clear identifyable target that specified the start and end of a mouse
# trajectory. In their paper Banholzer et al. (2021) identify movement trajectories by using a mouse movement event as
# the start and a click or scroll event as the end. To some extend, it is reasonable to assume that this mostly represent
# trajectories with an ideal line between the start point and the end point, but that is not necessarily the case.
# In this study, no mouse events were recorded, so trajectories were split by pauses, which makes the assumption
# that a trajectory represents a goal-directed straight movement even more unlikely

acc_preds <- c(
  'mo_ep_mean_angle_mean', 'mo_ep_sd_angle_mean', 'mo_ep_mean_angle_sd', 'mo_ep_sd_angle_sd', 'mo_ep_mean_x_flips',
  'mo_ep_sd_x_flips', 'mo_ep_mean_y_flips', 'mo_ep_sd_y_flips'
)
length(acc_preds)

speed_preds <- c(
  'mo_ep_mean_speed_mean', 'mo_ep_sd_speed_mean', 'mo_ep_mean_speed_sd', 'mo_ep_sd_speed_sd',
  'mo_ep_mean_abs_accel_mean', 'mo_ep_sd_abs_accel_mean', 'mo_ep_mean_abs_accel_sd', 'mo_ep_sd_abs_accel_sd',
  'mo_ep_mean_abs_jerk_mean', 'mo_ep_sd_abs_jerk_mean', 'mo_ep_mean_abs_jerk_sd', 'mo_ep_sd_abs_jerk_sd'
)
length(speed_preds)

#####################################
#### Creating different datasets ####
#####################################

# there are many possible specifications to the datasets, which might produce different results in the later analysis
# we decided to vary selected specifications to get a "richer picture of the analysis results"
# the specifications concern (1) calculation of mouse usage features for different pause threshold, (2) transforming
# the features to better fit to a normal distribution (3) the standardization of the input data

# (1) We choose three pause thresholds that separate mouse movement episodes from another, 1 second, 2 second and
#     3 seconds. The numbers were randomly chosen based on intuition

# (2) Because most calculated mouse usage features have a long tail (e.g. few trials had a lot more mouse movements
#     than average, we choose to transform the mouse usage data to better resemble a normal distribution. We choose
#     a yeo-johnson transformation, because it is able to handle 0 values in contrast to log transformation or box-cox
#     transformation

# (3) Regarding standardization, we chose two options: grand-sample standardization, participant-sample standardization
#     standardization was chosen over centering because it makes it easier to compare the input variables, which are
#     measured on very different scales

# The combination of the pause threshold values (3) and the standarization procedures (2) result in 6 different datasets
# Note, that there are many more options that could potentially be tested, but adding an option at least doubles the
# ammount of datasets that need to be tested (at least 12 datasets), which will increase computation time and
# interpretability of the results (many analysis are performed with each dataset)

### helper functions for the dataset preprocessing procedure ###

# simple helper to standardize across a selected list of variables
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

# simple helper to apply the yeo-johnson transformation to a selected set of variables
yeo_johnson_transform <- function (data, vars) {

  data <- data %>%
    # mutate the selected columns using the bestNormalize yeo-johnson transformer
    mutate(across(.cols = all_of(vars),
                  ~ bestNormalize::yeojohnson(.x, standardize = F)$x.t))
  # return the mutated dataframe
  return(data)
}

# initialize the dataset list that will iterated in the data analysis step
dataset_list <- list()

# create the combination of all data preprocessing options: the outlier removal procedure with 2 different threshold
# for the iqr removal procedure & the sample that the preprocessing is applied to (the entire sample or per participant)
pause_thresholds <- c("1000", "2000", "3000")
transform_use <- c("by_sample", "by_participant")

# create the combination of all pause threshold and standardization options
data_preprocessing_combinations <- expand.grid(pause_thresholds, transform_use)

for (row in seq_len(nrow(data_preprocessing_combinations))) {

  # get the raw data set
  data_to_transform <- mouse_data

  # get the preprocessing options
  p_thresh <- data_preprocessing_combinations$Var1[row]
  std_opt <- data_preprocessing_combinations$Var2[row]

  print(paste0("Creating Dataset for pause threshold ", p_thresh, "ms with standardization approach: ", std_opt))

  # select the relevant ivs for the specified pause threshold (add the pause threshold and X to the name, if it exists
  # in the column names vector of the dataframe (the lockscreen time variables have no pause threshold prefix)
  sel_ivs <- ifelse(is.element(paste0('X', p_thresh, "_", ivs), colnames(data_to_transform)),
                    paste0('X', p_thresh, "_", ivs), ivs)

  # create the relevant dataset (most of the following code could be done in one pipeline. I prefer to separate it
  # for better readability of the steps. This preference likely is not very R-like)

  # Step 1: select the relevant iv columns based on the pause threshold and get rid of all other pause threshold ivs
  data_to_transform <- data_to_transform %>%
    # rename the selected columns to the iv names
    rename_with(~ ivs[which(sel_ivs == .x)], .cols = sel_ivs) %>%
    # filter out all non selected/renamed iv columns
    select(!matches("1000|2000|3000"))

  # Step 2: Transform the ivs using the yeo-johnson transformation
  data_to_transform <- data_to_transform %>%
    yeo_johnson_transform(., ivs)

  # Step 3: Standardize the covariates (timestamp by person, all other covariates by sample)
  data_to_transform <- data_to_transform %>%
    standardize_across_variables(., covariates[!  covariates %in% "timestamp"]) %>%
    # now group by person and standardize the timestamp
    group_by(ID) %>%
    standardize_across_variables(., "timestamp") %>%
    ungroup()

  # Step 4: Standardize the ivs based on the selected standardization procedure (by participant or by sample)
  if (std_opt == "by_sample") {
    data_to_transform <- data_to_transform %>%
      # standardize across all the independent variables using the entire sample as the standardization reference
      standardize_across_variables(., ivs)
    # if the preprocessing is done by participant
  } else if (std_opt == "by_participant") {
    data_to_transform <- data_to_transform %>%
      # standardize the ivs by participant
      group_by(ID) %>%
      standardize_across_variables(., ivs) %>%
      ungroup()
  } else {
    stop("Something went wrong, data preprocessing stopped")
  }

  # Step 5: Remove highly correlated features from the dataset
  corr_table <- data_to_transform %>% select(all_of(ivs)) %>% cor(.)
  # find the index of all bivariate correlations above the specified threshold using the caret::findCorrelation func
  # https://www.rdocumentation.org/packages/caret/versions/6.0-90/topics/findCorrelation
  index <- caret::findCorrelation(corr_table, .80, exact = FALSE)
  # get the names of the columns that will be removed
  to_be_removed <- colnames(corr_table)[index]
  # remove the columns from the dataframe
  data_to_transform <- data_to_transform %>%  select(-all_of(to_be_removed))

  # Step 6: add the transformed dataset to the list of all datasets
  dataset_list[[paste0('p_thresh_', p_thresh, "_", std_opt)]] <- data_to_transform

}


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
  pdf(paste0('Free_',filename,".pdf"),
      width = 10, height = 8,
      bg = "white",
      colormodel = "cmyk",
      paper = "a4r")

  # get seperate diagnostic plots depending if its a logistic regression or "regular" linear regression
  # if its a binomial model (= logistic regression)
  if (family(model)[[1]] == "binomial") {

    # standard plot
    simulated_residuals <- simulateResiduals(fittedModel = model, plot = F)
    # recalculate the residuals with a grouping variable, standard plot might be misleading for binomial data
    # see https://cran.r-project.org/web/packages/DHARMa/vignettes/DHARMa.html#binomial-data
    grouped_residuals <- recalculateResiduals(simulated_residuals, group = data$ID)

    # save the first plot
    plot(simulated_residuals, title = "Standard Res Diagnostic Plot")
    # save the second plot
    plot(grouped_residuals, title = "Grouped Res Diagnostic Plot")

    dev.off()

  } else {
    # if its a "regular" linear model

    # model diagnostics visualization
    diagnostic_plots.1 <- DHARMa::simulateResiduals(fittedModel = model, plot = F)
    diagnostic_plots.2 <- sjPlot::plot_model(model, type='diag')

    # save the first plot
    plot(diagnostic_plots.1)
    # save the second plot
    grid.arrange(diagnostic_plots.2[[1]], diagnostic_plots.2[[2]]$ID, diagnostic_plots.2[[3]], diagnostic_plots.2[[4]], nrow = 2)

    dev.off()

  }
}


# function to create and save plots about the relationship between a single predictor variable and the target
# and taking the multilevel structure into account
# all Plots are marginal effect plot of the predictor variable on the target variable controlling for the covariates
plot_single_pred_mixed_model <- function (dataset, predictor, target, fixed_effect_model, random_effect_model, mod_family) {

  print("Plotting the Single Predictor model")

  # First, plot the relationship between predictor and target when ignoring the group structure plus individual
  # relationship plots per participant (individual regression models)

  # calculate the marginal effects between target and predictor per participant
  marg_effect_predictions <- dataset%>%
    group_by(ID) %>%
    do(preds = ggpredict(glm(data = ., formula = as.formula(paste0(target, " ~ ", predictor, " + ", paste(covariates, collapse = "+"))),
                             family = mod_family), terms = predictor))

  # create a single dataset for plotting from the predicted marginal effects per group and add a group label
  merged_marg_eff_preds <- bind_rows(marg_effect_predictions$preds, .id = "par")

  # plot the invididual regression lines
  lm_plot <- ggplot(data = merged_marg_eff_preds, aes(x = x, y = predicted, colour = par)) +
    # add participant regression line
    geom_line(size = 1, alpha = 0.5) +
    # add the regression line of the total dataset when ignoring the grouped data structure
    geom_line(data = ggpredict(glm(formula = as.formula(paste(target, " ~ ", predictor, " + ", paste(covariates, collapse = "+"))),
                                   data = dataset, family = mod_family), terms = predictor),
              aes(x = x, y = predicted), colour = "black", size = 2) +
    # add the conf intervals of the regression line
    geom_ribbon(data = ggpredict(glm(formula = as.formula(paste(target, " ~ ", predictor, " + ", paste(covariates, collapse = "+"))),
                                     data = dataset, family = mod_family), terms = predictor),
                aes(x = x, y = predicted, ymin = conf.low, ymax = conf.high, colour = NULL), alpha =0.2) +
    theme_light() +
    xlab(predictor) + ylab(target) +
    labs(title = paste0(target, " by ", predictor),
         subtitle = "Model per Group and Overall") +
    theme(legend.position="none")

  # Second, Plot the random intercept model with individual slopes per participant + the general slope of the random
  # intercept model
  ri_plot <- ggplot(data = ggpredict(fixed_effect_model, terms = c(predictor)), aes(x = x, y = predicted)) +
    # linear model for each participant with unique intercept
    geom_line(data = ggpredict(fixed_effect_model, terms = c(predictor, "ID"), type = "re"),
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
  rs_plot <- ggplot(data = ggpredict(random_effect_model, terms = c(predictor)), aes(x = x, y = predicted)) +
    # linear relationships for each participant
    geom_line(data = ggpredict(random_effect_model, terms = c(predictor, "ID"), type = "re"),
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
  save_path <- paste0("Free_single_pred_", predictor, '_', target, ".pdf")
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

# use ggpredict to create marginal effect plots of the interaction effects
plot_mixed_model_interaction <- function (predictor1, predictor2, target, fixed_effect_model, random_effect_model) {

  print("Plotting the interaction model")
  # plot the fixed effect interaction of the random intercept model
  ri_fixed_int <- ggpredict(fixed_effect_model, c(predictor1, predictor2))
  # plot the random intercept interactions for a subset of randomly selected participants (there are too many
  # participants for a single plot
  ri_rand_int <- ggpredict(fixed_effect_model, c(predictor1, predictor2, "ID [sample=6]"), type = "re")

  # plot the fixed effect interaction of the random intercept + random slope model
  rs_fixed_int <- ggpredict(random_effect_model, c(predictor1, predictor2))
  # plot the random intercept interactions for a subset of randomly selected participants (there are too many
  # participants for a single plot
  rs_rand_int <- ggpredict(random_effect_model, c(predictor1, predictor2, "ID [sample=6]"), type = "re")

  # save the plots as a pdf file with a plot on a seperate page
  save_path <- paste0("Free_interaction_", predictor1, '&', predictor2, '_', target, ".pdf")
  pdf(save_path,
      width = 10, height = 8,
      bg = "white",
      colormodel = "cmyk",
      paper = "a4r")

  # save the random intercept fixed effect interaction
  print(plot(ri_fixed_int) + ggtitle("Fixed Effects Interaction of Random Intercept Model"))
  # save the random intercept random effect interaction
  print(plot(ri_rand_int) + ggtitle("Random Effects Interaction of Random Intercept Model"))
  # save the random intercept + slope fixed effect interaction
  print(plot(rs_fixed_int) + ggtitle("Fixed Effects Interaction of Random Intercept & Slope Model"))
  # save the random intercept + slope random effect interaction
  print(plot(rs_rand_int) + ggtitle("Random Effects Interaction of Random Intercept & Slope Model"))

  dev.off()

}

#############################################################
### Helper Functions to fit different mixed effect models ###
#############################################################

# random intercept model
# the functiom has the option to enable/disable creating a model diagnostics plot, the diagnostic plots should be
# inspected for each model. Disabling it can save computational time
random_intercept_model <- function (dataset, target, model_family, plot_diag = F) {

  # test a random intercept model with the target variable
  # use a logistic regression or "regular" linear regression depending on the model_family
  # To simplify the syntax, instead of an if, else, a glmer model could be specficied with the model_family as the
  # family parameter, however, this gives a warning that a glmer model with family gaussian defaults to an lmer model
  # which uses REML (which is unwanted here)
  ri_form <- paste(target, '~', '1 + (1|ID)')
  if (model_family == "gaussian") {
   ri_model <- lmer(formula = as.formula(ri_form), data = dataset, REML = F, control=lmerControl(optimizer="bobyqa"))
  } else if (model_family == "binomial") {
    ri_model <- glmer(formula = as.formula(ri_form), data = dataset, family = 'binomial',
                      control=glmerControl(optimizer="bobyqa"))
  } else {
    stop("Model Family not specified.")
  }

  # model diagnostics visualization
  if (plot_diag) {model_diagnostic_plots(model = ri_model, filename = paste0("Task_intercept_only_", target), data = dataset) }

  # calculate the model coefficients
  coefficients <- broom.mixed::tidy(ri_model, conf.int = T, conf.method = "Wald")

  # calculate the Intra-Class-Correlation (comes with the model diagnostic criterias)
  model_diag <- performance::model_performance(ri_model)

  # return the model coefficients and the model ICC in a list
  list(coeffs = coefficients, model_diag = model_diag)

}

# baseline model
baseline_model <- function (dataset, target, model_family, plot_diag = F) {

  # setup the baseline model formular = predict the target with the control variables and the random intercept
  bs_form <- paste(target, '~', paste(covariates, collapse = "+"), ' + (1|ID)')
  if (model_family == "gaussian") {
    baseline_model <- lmer(formula = as.formula(bs_form), data = dataset, REML = F, control=lmerControl(optimizer="bobyqa"))
  } else if (model_family == "binomial") {
    baseline_model <- glmer(formula = as.formula(bs_form), data = dataset, family = 'binomial',
                      control=glmerControl(optimizer="bobyqa"))
  } else {
    stop("Model Family not specified.")
  }

  # model diagnostics visualization
  if (plot_diag) {model_diagnostic_plots(model = baseline_model, filename = paste0("Baseline_mod_", target), data = dataset) }

  # calculate the model coefficients
  coefficients <- broom.mixed::tidy(baseline_model, conf.int = T, conf.method = "Wald")

  # calculate the Intra-Class-Correlation (comes with the model diagnostic criterias)
  model_diag <- performance::model_performance(baseline_model)

  # return the model coefficients and the model ICC in a list
  list(coeffs = coefficients, model_diag = model_diag)

}

# single predictor model
single_predictor_model <- function (dataset, target, predictor, model_family, plot_diag = F, plot_model = F) {

  # fixed effect model
  print("Fitting the Fixed Effect Model")
  fe_form <- paste(target, '~', predictor, '+',paste(covariates, collapse = "+"), '+ (1|ID)')
  if (model_family == "gaussian") {
    fixed_effect_model <- lmer(formula = as.formula(fe_form), data = dataset, REML = F, control=lmerControl(optimizer = "bobyqa"))
  } else if (model_family == "binomial") {
    fixed_effect_model <- glmer(formula = as.formula(fe_form), data = dataset, family = 'binomial',
                                 control=glmerControl(optimizer="bobyqa"))
  } else {
    stop("Model Family not specified.")
  }
  fixed_effects_model_coeffs <- broom.mixed::tidy(fixed_effect_model, conf.int = T, conf.method = "Wald")
  fixed_effects_model_diagnostics <- performance::model_performance(fixed_effect_model)
  # create model diagnostic plots of the fixed effect model
  if (plot_diag) {model_diagnostic_plots(model = fixed_effect_model, filename = paste0("fe_single_pred_", predictor, "_", target),
                                         data = dataset)}

  # random effect model
  print("Fitting the Random Effect Model")
  re_form <- paste(target, '~', predictor, '+', paste(covariates, collapse = "+"), '+ (1 + ', predictor, '|ID)')
  if (model_family == "gaussian") {
    random_effect_model <- lmer(formula = as.formula(re_form), data = dataset, REML = F, control=lmerControl(optimizer = "bobyqa"))
  } else if (model_family == "binomial") {
    random_effect_model <- glmer(formula = as.formula(re_form),
                                 data = dataset, family = 'binomial', control=glmerControl(optimizer="bobyqa"))
  } else {
    stop("Model Family not specified.")
  }
  random_effects_model_coeffs <- broom.mixed::tidy(random_effect_model, conf.int = T, conf.method = "Wald")
  # for some cases, a really low variance value of the random effect will be estimated, which causes a singularity
  # warning: lower the threshold to silence the warning in most cases, but not in all
  # (-> likely not caused due to an overcomplex model, but due to the random effect being non-existent)
  random_effects_model_diagnostics <- performance::model_performance(random_effect_model, tolerance = 1e-15)
  # create model diagnostic plots of the random effects model
  if (plot_diag) {model_diagnostic_plots(model = random_effect_model, filename = paste0("re_single_pred_", predictor, "_", target),
                                         data = dataset)}

  # save plots of the single predictor model (this can take some time and is disabled if not specified)
  if (plot_model) {
    plot_single_pred_mixed_model(dataset = dataset, predictor = predictor, target = target,
                                 fixed_effect_model = fixed_effect_model, random_effect_model = random_effect_model,
                                 mod_family = model_family)
  }

  # return the main results of the single predictor model analysis in a list
  list(fe_coeffs_sp = fixed_effects_model_coeffs, fe_diag_sp = fixed_effects_model_diagnostics,
       re_coeffs_sp = random_effects_model_coeffs, re_diag_sp = random_effects_model_diagnostics
  )

}


# interaction model
interaction_model <- function (dataset, target, predictor1, predictor2, model_family, plot_diag = F, plot_model = F) {

  # fixed effect interaction model with both target variables as well as the interaction term
  print("Fitting the Fixed Effect Model")
  fe_form <- paste(target, '~', predictor1, '*', predictor2, '+', paste(covariates, collapse = "+"), '+ (1|ID)')
   if (model_family == "gaussian") {
    fixed_effect_model <- lmer(formula = as.formula(fe_form),
                             data = dataset, REML = F, control=lmerControl(optimizer = "bobyqa"))
  } else if (model_family == "binomial") {
    fixed_effect_model <- glmer(formula = as.formula(fe_form),
                                data = dataset, family = 'binomial',  control=glmerControl(optimizer="bobyqa"))
  } else {
    stop("Model Family not specified.")
  }
  fixed_effects_model_coeffs <- broom.mixed::tidy(fixed_effect_model, conf.int = T, conf.method = "Wald")
  fixed_effects_model_diagnostics <- performance::model_performance(fixed_effect_model)
  # create model diagnostic plots of the fixed effect model
  if (plot_diag) {model_diagnostic_plots(model = fixed_effect_model,
                                         filename = paste0("fe_interaction_", predictor1, '&', predictor2, "_", target),
                                         data = dataset)}

  # random effect interaction model with both target variables as well as the interaction term
  print("Fitting the Random Effect Model")
  re_form <- paste(target, '~', predictor1, '*', predictor2, '+', paste(covariates, collapse = "+"), '+ (1 +', predictor1, ':', predictor2,  '|ID)')
  if (model_family == "gaussian") {
    random_effect_model <- lmer(formula = as.formula(re_form),
                             data = dataset, REML = F, control=lmerControl(optimizer = "bobyqa"))
  } else if (model_family == "binomial") {
    random_effect_model <- glmer(formula = as.formula(re_form),
                                data = dataset, family = 'binomial',  control=glmerControl(optimizer="bobyqa"))
  } else {
    stop("Model Family not specified.")
  }
  random_effects_model_coeffs <- broom.mixed::tidy(random_effect_model, conf.int = T, conf.method = "Wald")
  # for some cases, a really low variance value of the random effect will be estimated, which causes a singularity
  # warning: lower the threshold to silence the warning in most cases, but not in all
  # (-> likely not caused due to an overcomplex model, but due to the random effect being non-existent)
  random_effects_model_diagnostics <- performance::model_performance(random_effect_model, tolerance = 1e-15)
  # create model diagnostic plots of the random effects model
  if (plot_diag) {model_diagnostic_plots(model = random_effect_model,
                                         filename = paste0("re_interaction_", predictor1, '&', predictor2, "_", target),
                                         data = dataset)}


  # save plots of the interaction models (this can take some time and is disabled if not specified)
  if (plot_model) {
    plot_mixed_model_interaction(predictor1, predictor2, target, fixed_effect_model, random_effect_model)
  }

  # return the main results of the interaction model analysis in a list
  list(fe_coeffs_int = fixed_effects_model_coeffs, fe_diag_int = fixed_effects_model_diagnostics,
       re_coeffs_int = random_effects_model_coeffs, re_diag_int = random_effects_model_diagnostics
  )
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
get_sample_data <- function (dset = NULL, target = NULL, predictor = NULL, interaction_pair = NULL) {

  # draw a random dataset if no dataset is specified
  play_dataset <- if (!is.null(dset)) dset else dataset_list[sample(seq_along(dataset_list), 1)]
  # draw a random target variable
  play_target <- if (!is.null(target)) target else sample(dvs, 1)
  # draw a random iv from the sample dataset
  play_pred <- if (!is.null(predictor)) target else sample(ivs[ivs %in% colnames(play_dataset[[1]])], 1)
  # get a random interaction pair
  if (!is.null(interaction_pair)) {
    play_interaction <- interaction_pair
  } else {
    # in order to get an interaction pair, the remaining acc ancuracy and speed predictors need to be filtered out from
    # all potential acc and speed preds, then randomly draw one acc and one speed pred
    play_acc_pred <- sample(acc_preds[acc_preds %in% colnames(play_dataset[[1]])], 1)
    play_speed_pred <- sample(speed_preds[speed_preds %in% colnames(play_dataset[[1]])], 1)
    play_interaction <- list("acc_pred" = play_acc_pred, "speed_pred" = play_speed_pred)
  }
  # model family
  play_family <- if (play_target == "stress") "binomial" else "gaussian"

  list("dset" = play_dataset, "target" = play_target, "pred" = play_pred, "interaction"=play_interaction, "mod_fam"=play_family)

}

# get a random dataset
rng_play_dat <- get_sample_data()

# Try out the model functions with the randomly generated data

# icc model results (create and save a model fit plot)
play_icc_mod_results <-  random_intercept_model(dataset = rng_play_dat[["dset"]][[1]],
                                                target = rng_play_dat[["target"]],
                                                model_family = rng_play_dat[["mod_fam"]],
                                                plot_diag = T)

# baseline model results
play_baseline_results <- baseline_model(dataset = rng_play_dat[["dset"]][[1]],
                                                target = rng_play_dat[["target"]],
                                                model_family = rng_play_dat[["mod_fam"]],
                                                plot_diag = T)

# single predictor model results (with plots)
play_sing_pred_res <- single_predictor_model(dataset = rng_play_dat[["dset"]][[1]],
                                             target = rng_play_dat[["target"]],
                                             predictor = rng_play_dat[["pred"]],
                                             model_family = rng_play_dat[["mod_fam"]],
                                             plot_diag = T, plot_model = T)

# interaction model results (with plots)
play_inter_results <- interaction_model(dataset = rng_play_dat[["dset"]][[1]],
                                        target = rng_play_dat[["target"]],
                                        predictor1 = rng_play_dat[["interaction"]][["acc_pred"]],
                                        predictor2 = rng_play_dat[["interaction"]][["speed_pred"]],
                                        model_family = rng_play_dat[["mod_fam"]],
                                        plot_diag = T, plot_model = T)

###############################
### Model Calculation Loops ###
###############################
# This section includes the loops to calculate all specified models using the data analysis helper functions
# Note that each loop involves multiple model calculations and can take some time.
# See the "Playground" section for options to calculate single models for selected dataframes, model specifications
# and variables

# simple helper function to save the model results as separate csv files
save_model_results <- function (model_list) {
  # loop the model results list and save the results
  lapply(names(model_list), function (x) write.csv(x=model_list[[x]], file = paste0(x, ".csv"), row.names = F))
}

####################################
### Random Intercept Model Loops ###
####################################

# setup a list to save the data from the variable each loop
# TODO: Writing out all code is tideous, prone for errors and not flexible. Could be replaced by a smarter solution (e.g a loop)
ri_coefficients_list <- list()
ri_model_diagnostics_list <- list()

# loop over all dataframes
for (i in seq_along(dataset_list)) {
  name <- names(dataset_list[i])
  dframe <- dataset_list[[i]]
  # loop over all dependent variables
  for (dv in dvs) {
    print(paste0("Random Intercept Model for target: ", dv, " and dataset: ", name))
    model_family <- if (dv == "stress") "binomial" else "gaussian"
    # calculate the random intercept model results
    ri_results <- random_intercept_model(dframe, dv, model_family = model_family, plot_diag = F)
    # extract the results and add information about the dataset and the dv to them
    coeffs <- ri_results[["coeffs"]] %>% mutate(dv = dv, dframe = name)
    diag <- ri_results[["model_diag"]] %>% mutate(dv = dv, dframe = name)
    # add the coefficient results to the list
    ri_coefficients_list[[paste0(name, "_", dv)]] <- coeffs
    # add the model diagnostic results to the list
    ri_model_diagnostics_list[[paste0(name, "_", dv)]] <- diag

  }
}

# convert each list to a "final results" dataframe and save the final results in a list
ri_results <- list("Free_Mouse_results_ri_coeffs" = dplyr::bind_rows(ri_coefficients_list),
                   "Free_Mouse_results_ri_diag" = dplyr::bind_rows(ri_model_diagnostics_list))

# save the results as csv files
save_model_results(ri_results)

# If already Calculated: Import the results from the CSV files instead of running the loop
ri_results <- list(
   "Free_Mouse_results_ri_coeffs" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Random_Intercept/Free_Mouse_results_ri_coeffs.csv"),
  "Free_Mouse_results_ri_diag" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Random_Intercept/Free_Mouse_results_ri_diag.csv")
)

############################
### Baseline Model Loops ###
############################

baseline_coefficients_list <- list()
baseline_model_diagnostics_list <- list()

# loop over all dataframes
for (i in seq_along(dataset_list)) {
  name <- names(dataset_list[i])
  dframe <- dataset_list[[i]]
  # loop over all dependent variables
  for (dv in dvs) {
    print(paste0("Baseline Model for target: ", dv, " and dataset: ", name))
    model_family <- if (dv == "stress") "binomial" else "gaussian"
    # calculate the random intercept model results
    baseline_results <- baseline_model(dframe, dv, model_family = model_family, plot_diag = F)
    # extract the results and add information about the dataset and the dv to them
    coeffs <- baseline_results[["coeffs"]] %>% mutate(dv = dv, dframe = name)
    diag <- baseline_results[["model_diag"]] %>% mutate(dv = dv, dframe = name)
    # add the coefficient results to the list
    baseline_coefficients_list[[paste0(name, "_", dv)]] <- coeffs
    # add the model diagnostic results to the list
    baseline_model_diagnostics_list[[paste0(name, "_", dv)]] <- diag

  }
}

# convert each list to a "final results" dataframe and save the final results in a list
baseline_results <- list("Free_results_baseline_coeffs" = dplyr::bind_rows(baseline_coefficients_list),
                   "Free_results_baseline_diag" = dplyr::bind_rows(baseline_model_diagnostics_list))

# save the results as csv files
save_model_results(baseline_results)

# If already Calculated: Import the results from the CSV files instead of running the loop
baseline_results <- list(
   "Free_results_baseline_coeffs" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Baseline/Free_results_baseline_coeffs.csv"),
  "Free_results_baseline_diag" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Baseline/Free_results_baseline_diag.csv")
)

###################################
### Single Predictor Model Loop ###
###################################

#TODO: This takes very long to process and should probably be parallized

# setup lists to save all results from the single predictor model analysis
single_pred_fixed_effect_coeff_list <- list()
single_pred_fixed_effect_diag_list <- list()
single_pred_random_effect_coeff_list <- list()
single_pred_random_effect_diag_list <- list()

# loop over all dataframes
for (i in seq_along(dataset_list)) {
  name <- names(dataset_list[i])
  dframe <- dataset_list[[i]]
  # get all ivs that remained in the selected dataframe after removal of collinear features
  remaining_ivs <- ivs[ivs %in% colnames(dframe)]
  # loop over the specified dependent variables
  for (dv in dvs) {
    model_family <- if (dv == "stress") "binomial" else "gaussian"
    # loop over all remaining independent variables
    for (iv in remaining_ivs) {
      print(paste0("Singple Pred Model for target: ", dv, "; pred: ", iv, "; and dataset: ", name))
      # get the results of the single predictor model analysis
      sing_pred_results <- single_predictor_model(dataset = dframe, target = dv, predictor = iv,
                                                  model_family = model_family, plot_diag = F, plot_model = F)
      # extract the results, add relevant info to them
      sp_fe_coff <- sing_pred_results[["fe_coeffs_sp"]] %>% mutate(dv = dv, iv = iv, dframe = name)
      sp_fe_diag <- sing_pred_results[["fe_diag_sp"]] %>% mutate(dv = dv, iv = iv, dframe = name)
      sp_re_coff <- sing_pred_results[["re_coeffs_sp"]] %>% mutate(dv = dv, iv = iv, dframe = name)
      sp_re_diag <- sing_pred_results[["re_diag_sp"]] %>% mutate(dv = dv, iv = iv, dframe = name)
      # add all results to the corresponding list
      single_pred_fixed_effect_coeff_list[[paste0(name, "_", iv, "_", dv)]] <- sp_fe_coff
      single_pred_fixed_effect_diag_list[[paste0(name, "_", iv, "_", dv)]] <- sp_fe_diag
      single_pred_random_effect_coeff_list[[paste0(name, "_", iv, "_", dv)]] <- sp_re_coff
      single_pred_random_effect_diag_list[[paste0(name, "_", iv, "_", dv)]] <- sp_re_diag
    }
  }
}

# convert each list to a "final results" dataframe and save the final results in a list
single_pred_results <- list(
  "Free_Mouse_results_sp_fe_coeffs" = dplyr::bind_rows(single_pred_fixed_effect_coeff_list),
  "Free_Mouse_results_sp_fe_diag" = dplyr::bind_rows(single_pred_fixed_effect_diag_list),
  "Free_Mouse_results_sp_re_coeffs" = dplyr::bind_rows(single_pred_random_effect_coeff_list),
  "Free_Mouse_results_sp_re_diag" = dplyr::bind_rows(single_pred_random_effect_diag_list)
)

# save the results as a csv
save_model_results(single_pred_results)

# If already Calculated: Import the results from the CSV files instead of running the loop
single_pred_results <- list(
  "Free_Mouse_results_sp_fe_coeffs" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Single_predictor/Free_Mouse_results_sp_fe_coeffs.csv"),
  "Free_Mouse_results_sp_fe_diag" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Single_predictor/Free_Mouse_results_sp_fe_diag.csv"),
  "Free_Mouse_results_sp_re_coeffs" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Single_predictor/Free_Mouse_results_sp_re_coeffs.csv"),
  "Free_Mouse_results_sp_re_diag" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Single_predictor/Free_Mouse_results_sp_re_diag.csv")
)


##############################
### Interaction Model Loop ###
##############################

#TODO: This takes very long to process and should probably be parallized

# setup lists to save all results from the single predictor model analysis
interaction_fixed_effect_coeff_list <- list()
interaction_fixed_effect_diag_list <- list()
interaction_random_effect_coeff_list <- list()
interaction_random_effect_diag_list <- list()

# loop over all dataframes
for (i in seq_along(dataset_list)) {
  name <- names(dataset_list[i])
  dframe <- dataset_list[[i]]
  # get all remaining accuracy and speed variables that remain in the dataset after multicollinear features are removed
  remain_acc_preds <- acc_preds[acc_preds %in% colnames(dframe)]
  remain_speed_preds <- speed_preds[speed_preds %in% colnames(dframe)]
  # create all possible interaction pairs between the accuracy-and speed-predictor and randomly draw 2 interaction
  # pairs for each speed variable (could also be the other way around, or choose the longer or shorter vector, here
  # there are more accuracy variables in the datasets than speed variables, so the drawn number of interactions
  # is smaller than
  all_interactions <- expand.grid(remain_acc_preds, remain_speed_preds)
  # randomly draw one interaction pair for each speed variable
   set.seed(123)
  interaction_speed_pairs <- all_interactions %>% group_by(Var2) %>% slice_sample(.data = ., n=1) %>% ungroup()
  # randomly draw one interaction pair for each accuracy variable
   set.seed(123)
  interaction_acc_pairs <- all_interactions %>% group_by(Var1) %>% slice_sample(.data = ., n=1) %>% ungroup()
  # bind the speed and accuracy pairs together and remove potential duplicate rows
  interaction_pairs <- dplyr::bind_rows(interaction_speed_pairs, interaction_acc_pairs) %>% distinct()
  # create the combinations of
  # loop over all dependent variables
  #TODO: Change to dvs to loop over all dependent variables
  for (dv in dvs) {
    model_family <- if (dv == "stress") "binomial" else "gaussian"
    # loop over all independent variable pairs
    for (i in seq(1, nrow(interaction_pairs))) {
      # get both predictors
      pred1 <- as.character(interaction_pairs$Var1[i])
      pred2 <- as.character(interaction_pairs$Var2[i])
      print(paste0("Interaction Model for target: ", dv, "; preds: ", pred1, " & ", pred2 ,"; and dataset: ", name))
      # get the results of the single predictor model analysis
      interaction_effect_results <- interaction_model(dataset = dframe, target = dv, predictor1 = pred1,
                                                      predictor2 = pred2, model_family = model_family,
                                                      plot_diag = F, plot_model = F)
      # extract the results, add relevant info to them
      int_fe_coff <- interaction_effect_results[["fe_coeffs_int"]] %>% mutate(dv = dv, iv1 = pred1, iv2 = pred2, dframe = name)
      int_fe_diag <- interaction_effect_results[["fe_diag_int"]] %>% mutate(dv = dv, iv1 = pred1, iv2 = pred2, dframe = name)
      int_re_coff <- interaction_effect_results[["re_coeffs_int"]] %>% mutate(dv = dv, iv1 = pred1, iv2 = pred2, dframe = name)
      int_re_diag <- interaction_effect_results[["re_diag_int"]] %>% mutate(dv = dv, iv1 = pred1, iv2 = pred2, dframe = name)
      # add all results to the corresponding list
      interaction_fixed_effect_coeff_list[[paste0(name, "_", pred1, "_", pred2, "_", dv)]] <- int_fe_coff
      interaction_fixed_effect_diag_list[[paste0(name, "_", pred1, "_", pred2, "_", dv)]] <- int_fe_diag
      interaction_random_effect_coeff_list[[paste0(name, "_", pred1, "_", pred2, "_", dv)]] <- int_re_coff
      interaction_random_effect_diag_list[[paste0(name, "_", pred1, "_", pred2, "_", dv)]] <- int_re_diag
    }
  }
}

# convert each list to a "final results" dataframe and save the final results in a list

interaction_results <- list(
  "Free_Mouse_results_interaction_fe_coeffs" = dplyr::bind_rows(interaction_fixed_effect_coeff_list),
  "Free_Mouse_results_interaction_fe_diags" = dplyr::bind_rows(interaction_fixed_effect_diag_list),
  "Free_Mouse_results_interaction_re_coeffs" = dplyr::bind_rows(interaction_random_effect_coeff_list),
  "Free_Mouse_results_interaction_re_diags" = dplyr::bind_rows(interaction_random_effect_diag_list)
)

# save all results as csv files
save_model_results(interaction_results)

# if the results are already calculated, just import them without doing the entire loop
interaction_results <- list(
  "Free_Mouse_results_interaction_fe_coeffs" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Interactions/Free_Mouse_results_interaction_fe_coeffs.csv"),
  "Free_Mouse_results_interaction_fe_diags" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Interactions/Free_Mouse_results_interaction_fe_diags.csv"),
  "Free_Mouse_results_interaction_re_coeffs" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Interactions/Free_Mouse_results_interaction_re_coeffs.csv"),
  "Free_Mouse_results_interaction_re_diags" = read.csv("Free_Mouse_Results/Mixed_Model_Results/Interactions/Free_Mouse_results_interaction_re_diags.csv")
)

#############################
### Visualize the Results ###
#############################

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
  ggsave(paste0(title,".png"), merged_plot, width = 22, height = 12)

  dev.off()

}

test <- interaction_results[["Free_Mouse_results_interaction_fe_coeffs"]] %>%
  # filter the relevant dv data
  filter(dv == "arousal") %>%
  # filter out all effects that are not plotted (only the fixed effects are plotted
  filter(.data = ., grepl(':', term))

for (i in unique(test$term)) {
  print(i)
}


# Save a plot for every dependent variable of the single predictor model results visualization
for (target in dvs) {
  # get the (cleaned) coefficient data for the random intercept only model
  fe_coeffs <- single_pred_results[["Free_Mouse_results_sp_fe_coeffs"]] %>%
    # filter the relevant dv data
    filter(dv == target) %>%
    # filter out all effects that are not plotted (only the fixed effects are plotted
    filter(.data = ., !grepl('(Intercept)|timestamp|zoom|screen_width|screen_height|sd_(Intercept)|sd__Observation', term)) %>%
    # rename the term values to better variable name values
    # BAD CODING: Renaming should have happened in an earlier step!
    mutate(term = recode(term,  "recording_duration"='Recording Duration',
                         "mo_ep_mean_episode_duration"='Move Ep. (mean): Episode Duration',
                         "mo_ep_mean_total_dist"='Move Ep. (mean): Tot. Distance',
                         "mo_ep_mean_speed_mean"='Move Ep. (mean): Speed (mean)',
                         "mo_ep_sd_speed_mean"='Move Ep. (sd): Speed (mean)',
                         "mo_ep_mean_angle_mean"='Move Ep. (mean): Angle (mean)',
                         "mo_ep_sd_angle_sd"='Move Ep. (sd): Angle (sd)',
                         "mo_ep_mean_x_flips"='Move Ep. (mean): X-Flips',
                         "no_movement"='No Movement',
                         "lockscreen_time"='Lockscreen Time',
                         "movement_episodes"='Num. of Move Ep.',
                         "mo_ep_sd_abs_jerk_mean"='Move Ep. (sd): Jerk (mean)',
                         "mo_ep_sd_angle_mean"='Move Ep. (sd): Angle (mean)',
                         "mo_ep_mean_angle_sd"='Move Ep. (mean): Angle (sd)',
                         "mo_ep_mean_speed_sd"='Move Ep. (mean): Speed (sd)',
                         "mo_ep_sd_total_dist"='Move Ep. (sd): Tot. Distance',
                         "mo_ep_mean_y_flips"='Move Ep. (mean): Y-Flips',
                         "lockscreen_episodes."='Num. of Lockscreen Eps.'
    ))

  # get the (cleaned) coefficient data for the random intercept + slope model
  re_coeffs <- single_pred_results[["Free_Mouse_results_sp_re_coeffs"]] %>%
    # filter the relevant dv data
    filter(dv == target) %>%
    # filter out all effects that are not plotted
    filter(.data = ., !grepl('(Intercept)|timestamp|zoom|screen_width|screen_height|sd_(Intercept)|sd__Observation', term)) %>%
    # remove the sd__ string from the random effect coefficient names in order to rename it in the next step
    mutate(term = str_replace(term, 'sd__', '')) %>%
    # rename the term values to better variable name values
    # BAD CODING: Renaming should have happened in an earlier step!
    mutate(term = recode(term,  "recording_duration"='Recording Duration',
                         "mo_ep_mean_episode_duration"='Move Ep. (mean): Episode Duration',
                         "mo_ep_mean_total_dist"='Move Ep. (mean): Tot. Distance',
                         "mo_ep_mean_speed_mean"='Move Ep. (mean): Speed (mean)',
                         "mo_ep_sd_speed_mean"='Move Ep. (sd): Speed (mean)',
                         "mo_ep_mean_angle_mean"='Move Ep. (mean): Angle (mean)',
                         "mo_ep_sd_angle_sd"='Move Ep. (sd): Angle (sd)',
                         "mo_ep_mean_x_flips"='Move Ep. (mean): X-Flips',
                         "no_movement"='No Movement',
                         "lockscreen_time"='Lockscreen Time',
                         "movement_episodes"='Num. of Move Ep.',
                         "mo_ep_sd_abs_jerk_mean"='Move Ep. (sd): Jerk (mean)',
                         "mo_ep_sd_angle_mean"='Move Ep. (sd): Angle (mean)',
                         "mo_ep_mean_angle_sd"='Move Ep. (mean): Angle (sd)',
                         "mo_ep_mean_speed_sd"='Move Ep. (mean): Speed (sd)',
                         "mo_ep_sd_total_dist"='Move Ep. (sd): Tot. Distance',
                         "mo_ep_mean_y_flips"='Move Ep. (mean): Y-Flips',
                         "lockscreen_episodes."='Num. of Lockscreen Eps.'
    ))
  # feed the datasets into the visualization function
  plot_coefficient_estimates(fe_coeffs, re_coeffs, paste0("Free_Mouse_single_predictor_estimates_for_target_", target),
                             dot_size = .75)

}

# plot the interaction effects
for (target in dvs) {
  # get the (cleaned) coefficient data for the random intercept only model
  fe_coeffs <- interaction_results[["Free_Mouse_results_interaction_fe_coeffs"]] %>%
    # filter the relevant dv data
    filter(dv == target) %>%
    # filter out all effects that are not plotted (only the interaction effects are plotted)
    filter(.data = ., grepl(':', term)) %>%
    # rename the term values to better variable name values
    # BAD CODING: Renaming should have happened in an earlier step!
    mutate(term = recode(term,  "mo_ep_mean_speed_mean:mo_ep_mean_angle_mean"='Move Ep. (Mean): Speed (mean) x Move Ep. (Mean): Angle (mean)',
                         "mo_ep_mean_speed_mean:mo_ep_sd_angle_sd"='Move Ep. (Mean): Speed (mean) x Move Ep. (sd): Angle (sd)',
                         "mo_ep_mean_speed_mean:mo_ep_mean_x_flips"='Move Ep. (mean): Speed (mean) x Move Ep. (Mean): X-Flips',
                         "mo_ep_sd_speed_mean:mo_ep_mean_x_flips"='Move Ep. (sd): Speed (mean) x Move Ep. (Mean): X-Flips',
                         "mo_ep_mean_speed_mean:mo_ep_sd_angle_mean"='Move Ep. (Mean): speed (mean) x Move Ep. (sd): Angle (mean)',
                         "mo_ep_mean_speed_mean:mo_ep_mean_angle_sd"='Move Ep. (Mean): Speed (mean) x Move Ep. (Mean): Angle (sd)',
                         "mo_ep_sd_abs_jerk_mean:mo_ep_mean_x_flips"='Move Ep. (sd): Jerk (mean) x Move Ep. (Mean): X-Flips',
                         "mo_ep_mean_speed_sd:mo_ep_sd_angle_mean"='Move Ep. (Mean): Speed (sd) x Move Ep. (sd): Angle (mean)',
                         "mo_ep_mean_speed_sd:mo_ep_mean_angle_sd"='Move Ep. (Mean): Speed (sd) x Move Ep. (Mean): Angle (sd)',
                         "mo_ep_mean_speed_sd:mo_ep_mean_x_flips"='Move Ep. (Mean): Speed (sd) x Move Ep. (Mean): X-Flips',
                         "mo_ep_mean_speed_mean:mo_ep_mean_y_flips"='Move Ep. (Mean): Speed (mean) x Move Ep. (Mean): Y-Flips',
                         "mo_ep_sd_abs_jerk_mean:mo_ep_mean_y_flips"='Move Ep. (sd): Jerk (mean) x Move Ep. (Mean): Y-Flips',
                         "mo_ep_sd_abs_jerk_mean:mo_ep_sd_angle_mean"='Move Ep. (sd): Jerk (mean) x Move Ep. (sd): Angle (mean)',
    ))

  # get the (cleaned) coefficient data for the random intercept + slope model
  re_coeffs <- interaction_results[["Free_Mouse_results_interaction_re_coeffs"]] %>%
    # filter the relevant dv data
    filter(dv == target) %>%
    # filter out all effects that are not plotted
    filter(.data = ., grepl(':', term) & !grepl('cor_', term)) %>%
    # remove the sd__ string from the random effect coefficient names in order to rename it in the next step
    mutate(term = str_replace(term, 'sd__', '')) %>%
    # rename the term values to better variable name values
    # BAD CODING: Renaming should have happened in an earlier step!
    mutate(term = recode(term,  "mo_ep_mean_speed_mean:mo_ep_mean_angle_mean"='Move Ep. (Mean): Speed (mean) x Move Ep. (Mean): Angle (mean)',
                         "mo_ep_mean_speed_mean:mo_ep_sd_angle_sd"='Move Ep. (Mean): Speed (mean) x Move Ep. (sd): Angle (sd)',
                         "mo_ep_mean_speed_mean:mo_ep_mean_x_flips"='Move Ep. (mean): Speed (mean) x Move Ep. (Mean): X-Flips',
                         "mo_ep_sd_speed_mean:mo_ep_mean_x_flips"='Move Ep. (sd): Speed (mean) x Move Ep. (Mean): X-Flips',
                         "mo_ep_mean_speed_mean:mo_ep_sd_angle_mean"='Move Ep. (Mean): speed (mean) x Move Ep. (sd): Angle (mean)',
                         "mo_ep_mean_speed_mean:mo_ep_mean_angle_sd"='Move Ep. (Mean): Speed (mean) x Move Ep. (Mean): Angle (sd)',
                         "mo_ep_sd_abs_jerk_mean:mo_ep_mean_x_flips"='Move Ep. (sd): Jerk (mean) x Move Ep. (Mean): X-Flips',
                         "mo_ep_mean_speed_sd:mo_ep_sd_angle_mean"='Move Ep. (Mean): Speed (sd) x Move Ep. (sd): Angle (mean)',
                         "mo_ep_mean_speed_sd:mo_ep_mean_angle_sd"='Move Ep. (Mean): Speed (sd) x Move Ep. (Mean): Angle (sd)',
                         "mo_ep_mean_speed_sd:mo_ep_mean_x_flips"='Move Ep. (Mean): Speed (sd) x Move Ep. (Mean): X-Flips',
                         "mo_ep_mean_speed_mean:mo_ep_mean_y_flips"='Move Ep. (Mean): Speed (mean) x Move Ep. (Mean): Y-Flips',
                         "mo_ep_sd_abs_jerk_mean:mo_ep_mean_y_flips"='Move Ep. (sd): Jerk (mean) x Move Ep. (Mean): Y-Flips',
                         "mo_ep_sd_abs_jerk_mean:mo_ep_sd_angle_mean"='Move Ep. (sd): Jerk (mean) x Move Ep. (sd): Angle (mean)',
    ))

  # feed the datasets into the visualization function
  plot_coefficient_estimates(fe_coeffs, re_coeffs, paste0("Free_Mouse_interaction_estimates_for_target_", target), dot_size = .75)

}

#################################
### Bayes Mixed Model Testing ###
#################################
# Compare the results with a Bayesian Mixed Model Approach
# -----------------------------------------------------------
# Banholzer et al. (2021) use bayesian mixed models to test their hypothesis about a relationship between mouse
# usage and stress. We used a frequentist approach mostly because of the increased compuational cost of bayesian mixed
# models. Here the results are calculated for a random single variable (variable pair) and can be compared with
# the results of the lme4 model results

# Helper Function to visually compare the model results of the brm model with the coressponding lme4 model
# unfortunately, the broom.mixed::tidy() function throws an error for the (most) bayesian models (not all), therefore
# use the alternative parameters() function to create a comparison plot. It requires that the lme4 model is calculated
# again
compare_brm_lme4 <- function (formular, brm_model, filename) {

  # first get the results of the brm and lme4 model in the same dataframe
  sp_brm_vs_lm4 <- bayes_dataset[["dset"]][[1]] %>%
    # calculate the corresponding lme4 model
    glmer(formula = formular, data = ., family = 'binomial', control=glmerControl(optimizer="bobyqa")) %>%
    # get the model parameters
    parameters(., effects = "all") %>%
    # mutate it
    mutate(model = "lme4") %>%
    # drop the SD (Oberservation) row
    filter(Parameter != "SD (Observations)") %>%
    # add the model results of the brm model
    bind_rows(parameters(brm_model, effects = "all") %>% mutate(model = "brms") %>% mutate(Coefficient = Median)) %>%
    # set equal names to the parameters for both models
    mutate(Parameter = c(.$Parameter[1:(nrow(.)/2)], .$Parameter[1:(nrow(.)/2)])) %>%
    # plot the results
    ggplot(., aes(x=Coefficient, y=Parameter, color=model, group=model)) +
      geom_vline(xintercept = 0, colour = "black", linetype = 2, size=1) +
      # plot the fixed effect coefficients with their CIs
      geom_pointrange(aes(xmin = CI_low, xmax = CI_high), position=position_dodge(width = 0.5), size=1) +
      labs(x="Coefficient Estimates", y="Coefficients") +
      # set a scale limit to be able to better compare the small effects, this excludes larger effect e.g. the
      # intercept and can be disables
      scale_x_continuous(limits = c(-10, 10)) +
      theme_minimal() +
      theme(text = element_text(size = 14), legend.text = element_text(size = 14))

  # save the resulting image in the working directory
  ggsave(paste0(filename, ".png"), sp_brm_vs_lm4, width = 10, height = 8)

}

# get a random dataset for comparing the results between the brm model and the lme4 model
# we specified arousal as the target (which can be changed, of course)
bayes_dataset <- get_sample_data(target = "stress")

# run a bayesian mixed model with random intercept (no random slope), just like Banholzer et al. (2021) did.

########################
# single predictor model
########################

# specify the formular
bayes_sp_formula <- paste(bayes_dataset[["target"]], '~ 0 + Intercept + ', bayes_dataset[["pred"]], '+', paste(covariates, collapse = "+"),
                          '+ (1|ID)')
print(paste0("Bayesian Mixed Random Intercept Model for target: ", bayes_dataset[["target"]], "; predictor: ",
             bayes_dataset[["pred"]],"; and dataset: ", names(bayes_dataset[["dset"]])))

# Set the same priors as Banholzer et al. (2021). Their prior choice followed recommendations from the Stan
# developer team, https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations

priors <- c(set_prior('student_t(7,0,2.5)', class = 'b'),
            set_prior('student_t(7,0,10)', class = 'b', coef = 'Intercept'),
            set_prior('normal(0,1)', class = 'sd'))

# calc the model depending on the model family
if (bayes_dataset[["mod_fam"]] == "gaussian") {
  # increase the number of iterations due to convergence warning and train the 4 chains on parallel cores
  bayes_sp_model <- brms::brm(data = bayes_dataset[["dset"]][[1]],
                              formula = bayes_sp_formula,
                              prior = priors,
                              cores = 6, seed = 123, iter = 3000, chains = 6)
} else {
  bayes_sp_model <- brms::brm(data = bayes_dataset[["dset"]][[1]],
                              formula = bayes_sp_formula,
                              cores = 6, family = bernoulli(link = "logit"), seed = 123, iter = 3000, chains = 6)
}

# get a model summary
summary(bayes_sp_model)

# create a plot to compare the results of the single predictor model
bay_sing_pred_title <- paste0("Free_brm_vs_lme4_", bayes_dataset[["target"]], "_", bayes_dataset[["pred"]])
glmer_sp_formular <- paste(bayes_dataset[["target"]], '~ ', bayes_dataset[["pred"]], '+', paste(covariates, collapse = "+"),
                          '+ (1|ID)')
compare_brm_lme4(glmer_sp_formular, bayes_sp_model, bay_sing_pred_title)

####################
# interaction model
####################

# do the same for an interaction effect model
# for this to work, we need to get an interaction effect, that was also calculated by the lme4 model (not all possible
# interaction effects were calculated), so we need to grab the corresponding accuracy and speed interaction variables
# from the lme4 model

# get a random interaction pair from the calculated interaction models with the given dataset and the given target
# variable
bayes_interaction_pair <- interaction_results[["Free_Mouse_results_interaction_fe_coeffs"]] %>%
  filter(dframe == names(bayes_dataset[["dset"]])) %>%
  filter(dv == bayes_dataset[["target"]]) %>%
  distinct(iv1, iv2) %>%
  sample_n(., 1)

# specify the interaction model formular
bayes_interaction_formula <- paste(bayes_dataset[["target"]], '~ 0 + Intercept + ', bayes_interaction_pair[[1]], '*',
                                   bayes_interaction_pair[[2]], '+', paste(covariates, collapse = "+"),
                                   '+ (1|ID)')
print(paste0("Bayesian Interaction Model for target: ", bayes_dataset[["target"]], "; preds: ",
             bayes_interaction_pair[[1]], " & ", bayes_interaction_pair[[2]] ,"; and dataset: ", names(bayes_dataset[["dset"]])))

# calc the model depending on the model family
if (bayes_dataset[["mod_fam"]] == "gaussian") {
  # increase the number of iterations due to convergence warning and train the 4 chains on parallel cores
  bayes_interaction_model <- brms::brm(data = bayes_dataset[["dset"]][[1]],
                              formula = bayes_interaction_formula, prior = priors,
                              cores = 6, seed = 123, iter = 3000, chains = 6)
} else {
  bayes_interaction_model <- brms::brm(data = bayes_dataset[["dset"]][[1]],
                              formula = bayes_interaction_formula, prior = priors,
                              cores = 6, family = bernoulli(link = "logit"), seed = 123, iter = 3000, chains = 6)
}


# get a model summary
summary(bayes_interaction_model)

# create a plot to compare the results of the interaction model
bay_interaction_title <- paste0("Free_Mouse_brm_vs_lme4_", bayes_dataset[["target"]], "_", bayes_interaction_pair[[1]], "_", bayes_interaction_pair[[2]])
glmer_int_form <- paste(bayes_dataset[["target"]], '~ ', bayes_interaction_pair[[1]], '*',
                                   bayes_interaction_pair[[2]], '+', paste(covariates, collapse = "+"),
                                   '+ (1|ID)')
compare_brm_lme4(glmer_int_form, bayes_interaction_model, bay_interaction_title)