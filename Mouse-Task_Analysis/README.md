### Mouse-Task Data Analysis Code, Datasets & Results

This folder contains all the data analysis code, the processed datasets that were used to analyze the
mouse-task data as well as the analysis results of the mouse-task analysis.

Specifically, this folder contains:

- The python code to process the raw mouse-task data and calculate the mouse-task features. The raw-data is needed to run this
file [Mouse_Task_Feature_Calculation.py]
- Python code to inspect the data quality of selected mouse-task cases (e.g. mouse usage data of cases with extreme values
on selected mouse-features are visualized). To run this file, the raw-data is needed as well as the Data Quality
Inspection dataset [Mouse_Task_Data_Quality_Analysis.py]
- R-code to run the mixed model analysis. To run this file, the Mouse-task Features dataset is needed.
To reproduce the calculated results of the mixed-model analysis (e.g. visualize
the fixed and random effects), the results of the mixed-model analysis are also provided in the results folder [Mouse_Task_Mixed_Model_Analysis.R]
- Python code to run the machine learning analysis. To run this file, the Mouse-Task Features dataset is needed.
To reproduce the calculated results of the machine learning analysis (e.g. visualize
the feature importance scores), the results of the machine learning are also provided in the results folder [Mouse_Task_Machine_Learning_Analysis.py]
- The dataset that was used for inspecting the data qualitiy of the mouse-task data [Mouse_Task_Features_Data_Quality_Inspection.csv]
- The dataset that was used to run the mixed-model analysis as well as the machine learning analysis [Mouse_Task_Features.csv]
- A folder that contains the results of the mixed-model analysis as well as the machine learning analysis [Result_Datasets_Mouse-Task]