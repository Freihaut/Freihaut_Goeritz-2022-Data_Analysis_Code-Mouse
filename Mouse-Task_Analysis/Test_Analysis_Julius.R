'
Test file for inspecting the analysis with Julius
'

# package import
library(lme4)
library(lmerTest)
library(parameters)
library(tidyverse)
library(sjstats)
library(effects)
library(broom.mixed)
library(ggeffects)
library(texreg)
library(extraoperators)
library(JWileymisc)
library(multilevelTools)
library(sjPlot)
library(sjmisc)
library(sjlabelled)

# -- Helper Functions for mixed model analysis -- #

# function to output model 

# get the model coefficients and model performance criteria
model_infos <- function(mod) {
  
  # calculate the model coefficients
  coefficients <- broom.mixed::tidy(mod, conf.int = T, conf.method = "Wald")
  
  # calculate the Intra-Class-Correlation (comes with the model diagnostic criterias)
  performance_criteria <- performance::model_performance(mod)
  
  list(coeffs = coefficients, performance = performance_criteria)
  
}

# Variance reduction approach to compare models according to Hox, 2010, pp. 69-78
var_reduction = function(m0, m1){
  VarCorr(m0) %>% 
    as.data.frame %>% 
    select(grp, var_m0 = vcov) %>% 
    left_join(VarCorr(m1) %>% 
                as.data.frame %>% 
                select(grp, var_m1 = vcov)) %>% 
    mutate(var_red = 1 - var_m1 / var_m0) 
}

# --- Data Analysis with Toy Data --- # 

# import toy data
data(aces_daily, package = "JWileymisc")

# get basic infos about the data

# -overall structure
str(aces_daily, nchar.max = 30)

# The data shows that participants were repeatedly measured on different 
# questions (e.g. regarding their Positive and negative affect and regarding
# their stress level). The data also includes sociodemographics. In the
# toy analysis, we will focus on the relationship between negative affect and
# stress (both level 1 variables)

# basic infos about the cluster
length(unique(aces_daily$UserID))

# - basic info about variables of interest
aces_daily %>%
  select(STRESS, NegAff) %>%
  summary()

# - basic histogram/density plot of the variables of interest

# stress
ggplot(aces_daily, aes(x=STRESS)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666") 

# Negative Affect
ggplot(aces_daily, aes(x=NegAff)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666") 


# -center the predictor variable
# most upcoming analysis are taken from:
# https://philippmasur.de/2018/05/23/how-to-center-in-multilevel-models/#easy-footnote-5-242

aces_daily <- aces_daily %>% 
  # center around the grand mean (do this for the predictor and the outcome)
  mutate(STRESS.grandMeanCent = STRESS-mean(STRESS, na.rm=TRUE),
         NegAff.grandMeanCent = NegAff-mean(NegAff, na.rm=TRUE)) %>%
  # Person mean centering (again, for the predictor and outcome)
  group_by(UserID) %>% 
    mutate(STRESS.persMean = mean(STRESS, na.rm=TRUE),
           STRESS.persMeanCent = STRESS-STRESS.persMean,
           NegAff.persMeanCent = NegAff - mean(NegAff, na.rm=TRUE)) %>%
  ungroup %>%
  # Grand mean centering of the aggregated variable
  mutate(STRESS.gmPersMeanCent = STRESS.persMean-mean(STRESS.persMean, na.rm=TRUE))

# get some info about the centered variables
aces_daily %>% select(STRESS, STRESS.grandMeanCent, STRESS.persMeanCent, 
                      STRESS.gmPersMeanCent, NegAff, NegAff.grandMeanCent,
                      NegAff.persMeanCent) %>% summary()


# - Start with some Model Building

# null model
m0 <- lmer(NegAff ~ 1 + (1| UserID), data = aces_daily)
model_infos(m0)

# fixed effect models (with different centered variables)
m1.orig <- lmer(NegAff ~ STRESS + (1| UserID), data = aces_daily)
model_infos(m1.orig)

m1.grandMean <- lmer(NegAff ~ STRESS.grandMeanCent + (1| UserID), data = aces_daily)
model_infos(m1.grandMean)

m1.personMean <- lmer(NegAff ~ STRESS.persMeanCent + (1| UserID), data = aces_daily)
model_infos(m1.personMean)

m1.personMean2 <- lmer(NegAff ~ STRESS.persMeanCent + STRESS.gmPersMeanCent + 
                         (1| UserID), data = aces_daily)
model_infos(m1.personMean2)

# comparison table of models
screenreg(list(m1.orig, m1.grandMean, m1.personMean, m1.personMean2), 
          single.row = T, 
          stars = numeric(0),
          caption = "",
          custom.note = "Model 1 = Uncentered predictor, 
                         Model 2 = grand-mean centered predictor, 
                         Model 3 = person-mean centered predictor, 
                         Model 4 = person-mean centered predictor and centered person mean")


# Do some variance decomposition analysis comparison
cbind(M1 = var_reduction(m0, m1.orig)[,4],
      M2 = var_reduction(m0, m1.grandMean)[,4],
      M3 = var_reduction(m0, m1.personMean)[,4],
      M4 = var_reduction(m0, m1.personMean2)[,4]) %>%
  round(2)

# now calc a model with a random slope
m2 <- lmer(NegAff ~ STRESS.persMeanCent + STRESS.gmPersMeanCent + 
             (1 + STRESS.persMeanCent| UserID), data = aces_daily)
model_infos(m2)
# The model has a much better AIC fit as compared to the model without
# random slope. However, the R²-values are worse?

# plot the effect of Stress variable on negative affect
ggpredict(m2, terms = c("STRESS.persMeanCent", "UserID [sample=9]"), type="re") %>% plot()

# - Build Models with differently scaled dependent variable and compare them
mDV.grandMean <- lmer(NegAff.grandMeanCent ~ STRESS.persMeanCent + STRESS.gmPersMeanCent + 
                        (1 + STRESS.persMeanCent| UserID), data = aces_daily)
model_infos(mDV.grandMean)

mDV.persMean <- lmer(NegAff.persMeanCent ~ STRESS.persMeanCent + STRESS.gmPersMeanCent + 
                       (1 + STRESS.persMeanCent| UserID), data = aces_daily)
model_infos(mDV.persMean)

# comparison table of models
screenreg(list(m2, mDV.grandMean, mDV.persMean), 
          single.row = T, 
          stars = numeric(0),
          caption = "",
          custom.note = "Model 1 = Uncentered DV, 
                         Model 2 = grand-mean centered DV, 
                         Model 3 = person-mean centered DV")

# one last model that has both, the dv as well as the iv person mean centered
persMeanCentMod <- lmer(NegAff.persMeanCent ~ STRESS.persMeanCent +
                          (0 + STRESS.persMeanCent| UserID), data = aces_daily)
model_infos(persMeanCentMod)

# plot the model to see if it works
ggpredict(persMeanCentMod, terms = c("STRESS.persMeanCent", "UserID [sample=9]"), type="re") %>% plot()


# -- Real Data -- #

# - setup

# clear environment
rm(list = ls())
# set the working directory to be able to read the data
setwd("C:/Users/Paul/Desktop/Data_Analysis/Mouse-Task_Analysis")

# import the data
real.data <- read.csv(file = 'Mouse_Task_Features.csv')

# again, get some basic infos about the data (copy & paste workflow from above)
str(real.data, nchar.max = 30)

# number of unique participants
length(table(real.data$ID))

# in this demo analysis, the variables of interest are arousal as the DV and
# task_total_dist as the IV (we can also add gender as a control in a later step) 

# - basic info about variables of interest
real.data %>%
  select(arousal, task_total_dist) %>%
  summary()

# - basic histogram/density plot of the variables of interest

# IV
ggplot(real.data, aes(x=task_total_dist)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666") 

# DV
ggplot(real.data, aes(x=arousal)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666")

# we skip further data preprocessing for the sake of this example (e.g. outlier
# removal, task difference considerations...


# do the centering
real.data <- real.data %>% 
  # center around the grand mean (do this for the predictor and the outcome)
  mutate(dist.grandMeanCent = task_total_dist-mean(task_total_dist, na.rm=TRUE),
         arousal.grandMeanCent = arousal-mean(arousal, na.rm=TRUE)) %>%
  # Person mean centering (again, for the predictor and outcome)
  group_by(ID) %>% 
  mutate(dist.persMean = mean(task_total_dist, na.rm=TRUE),
         dist.persMeanCent = task_total_dist-dist.persMean,
         arousal.persMeanCent = arousal - mean(arousal, na.rm=TRUE)) %>%
  ungroup %>%
  # Grand mean centering of the aggregated variable
  mutate(dist.gmPersMeanCent = dist.persMean-mean(dist.persMean, na.rm=TRUE))

# get some info about the centered variables
real.data %>% select(task_total_dist, dist.grandMeanCent, dist.persMeanCent, 
                     dist.gmPersMeanCent, arousal, arousal.grandMeanCent,
                     arousal.persMeanCent) %>% summary()

# start the model building

# null model
m0 <- lmer(arousal ~ 1 + (1| ID), data = real.data)
model_infos(m0)

# fixed effect models
# fixed effect models (with different centered variables)
m1.orig <- lmer(arousal ~ task_total_dist + (1| ID), data = real.data)
model_infos(m1.orig)

m1.grandMean <- lmer(arousal ~ dist.grandMeanCent + (1| ID), data = real.data)
model_infos(m1.grandMean)

m1.personMean <- lmer(arousal ~ dist.persMeanCent + (1| ID), data = real.data)
model_infos(m1.personMean)

m1.personMean2 <- lmer(arousal ~ dist.persMeanCent + dist.gmPersMeanCent
                       + (1| ID), data = real.data)
model_infos(m1.personMean2)

# comparison table of models
screenreg(list(m1.orig, m1.grandMean, m1.personMean, m1.personMean2), 
          single.row = T, 
          stars = numeric(0),
          caption = "",
          custom.note = "Model 1 = Uncentered predictor, 
                         Model 2 = grand-mean centered predictor, 
                         Model 3 = person-mean centered predictor, 
                         Model 4 = person-mean centered predictor and centered person mean")

# Do some variance decomposition analysis comparison
cbind(M1 = var_reduction(m0, m1.orig)[,4],
      M2 = var_reduction(m0, m1.grandMean)[,4],
      M3 = var_reduction(m0, m1.personMean)[,4],
      M4 = var_reduction(m0, m1.personMean2)[,4]) %>%
  round(2)


# random slope model
# now calc a model with a random slope
m2 <- lmer(arousal ~ dist.persMeanCent + dist.gmPersMeanCent + 
             (1 + dist.persMeanCent| ID), data = real.data)
model_infos(m2)
# The model has a much better AIC fit as compared to the model without
# random slope. However, the R²-values are worse?

# plot the effect of Stress variable on negative affect
ggpredict(m2, terms = c("dist.persMeanCent", "ID [sample=9]"), type="re") %>% plot()

# - Build Models with differently scaled dependent variable and compare them
mDV.grandMean <- lmer(arousal.grandMeanCent ~ dist.persMeanCent + dist.gmPersMeanCent + 
                        (1 + dist.persMeanCent| ID), data = real.data)
model_infos(mDV.grandMean)

mDV.persMean <- lmer(arousal.persMeanCent ~ dist.persMeanCent + dist.gmPersMeanCent + 
                       (1 + dist.persMeanCent| ID), data = real.data)
model_infos(mDV.persMean)

# comparison table of models
screenreg(list(m2, mDV.grandMean, mDV.persMean), 
          single.row = T, 
          stars = numeric(0),
          caption = "",
          custom.note = "Model 1 = Uncentered DV, 
                         Model 2 = grand-mean centered DV, 
                         Model 3 = person-mean centered DV")

# one last model that has both, the dv as well as the iv person mean centered
persMeanCentMod <- lmer(arousal.persMeanCent ~ dist.persMeanCent + dist.gmPersMeanCent + 
                          (0 + dist.persMeanCent| ID), data = real.data)
model_infos(persMeanCentMod)

# plot the model to see if it works
ggpredict(persMeanCentMod, terms = c("dist.persMeanCent", "ID [sample=9]"), type="re") %>% plot()
