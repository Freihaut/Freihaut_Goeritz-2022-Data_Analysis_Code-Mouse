[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7576822.svg)](https://doi.org/10.5281/zenodo.7576822)

## Study Data Analysis Code: Exploring the relationship between emotional states and computer mouse usage

### General Information

This repository contains all data analysis code, which was used to analyze the longitudinal field study data about
the relationship between computer mouse usage and emotional states. The repository does not contain the original raw
data, because the data files are too large for github. The raw data can be found here: https://doi.org/10.5281/zenodo.6559329

The corresponding research manuscript is: Freihaut, P. & Göritz, A. S. (2022). Show me how you use your mouse and I tell you how you feel? Sensing affect with the computer mouse. Manuscript in review

The data was analyzed with Python (3.8.12) and R (4.1.1)


### Repository Structure

The repository contains 3 folders

1. The **Contextless-Mouse Analysis Folder** contains all data analysis code, which was used to analyze the self-guided mouse usage
behavior recorded during 5-minutes of regular computer usage. The folder also contains the analysis results. Please see the Readme in the folder for a more detailed 
description.
2. The **Mouse-Task Analysis Folder** contains all data analysis code, which was used to analyze the mouse usage behavior
which was recorded during the standardized point-and-click task. The folder also contains all analysis results. Please see the Readme in the folder for a more detailed 
description.
3. The **General Analysis Folder** contains files that are not specific to either the analysis or the contextless-mouse data
or the analysis of the mouse-task data. Please see the Readme in the folder for a more detailed description.


In the folders you will find processed data sets in addition to the data analysis code. They can be used to run the
analysis code with the files that we used for data analysis. Please note that many functions in the analysis code
of the contextless-mouse data and mouse-task data are identical. From a coding perspective, it would make the code better
if general purpose functions would appear in a separate file. However, we deemed it easier to read and run the code if
the functions appeared in both analysis code files. 

For any questions regarding the data analysis code, contact: pfreihaut@gmail.com
