### Free-Mouse Data Analysis Code, Datasets & Results

This folder contains all the data analysis code, the processed datasets that were used to analyze the
free-mouse data as well as the analysis results of the mouse-task analysis.

Specifically, this folder contains:

- The python code to process the raw free-mouse data and calculate the free-mouse features. The raw-data is needed to run this
file [Free_Usage_Feature_Calculation.py]
- Python code to inspect the data quality of selected free-mouse cases (e.g. mouse usage data of cases with extreme values
on selected mouse-features are visualized). To run this file, the raw-data is needed as well as the Data Quality
Inspection dataset [Free_Usage_Data_Quality_Analysis.py]
- R-code to run the mixed model analysis. To run this file, the Free Mouse Features dataset is needed.
To reproduce the calculated results of the mixed-model analysis (e.g. visualize the fixed and random effects),
the results of the mixed-model analysis are provided in the results folder [Free_Mouse_Mixed_Model_Analysis.R]
- Python code to run the machine learning analysis. To run this file, the Free Mouse Features dataset is needed.
To reproduce the calculated results of the machine learning analysis (e.g. visualize
the feature importance scores), the results of the machine learning are provided in the results folder [Free_Usage_Machine_Learning_Analysis.py]
- The dataset that was used for inspecting the data qualitiy of the free-mouse data [Free_Mouse_Features_Data_Quality_Inspection.csv]
- The dataset that was used to run the mixed-model analysis as well as the machine learning analysis [Free_Mouse_Features.csv]
- A folder that contains the results of the mixed-model analysis as well as the machine learning analysis [Result_Datasets_Free-Mouse]