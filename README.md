# JTB-DNA_Binding

# The Datasets
The datasets folder contains all the feature for the experiments. All the features need to unzipped and kept in the datasets folder for the codes to run properly.
* "**All_32620_Features_Test_and_Train.zip**" contains all the features extratced from both the train and the test datasets, these were used for the Recursive feature Selection.
* "**Group Test Dataset.zip**" and "**Group Train Dataset.zip**" contains test and train files for the Grouped feature Selection. The features groups are separated in different csv files.


# The Codes

######  Grouped Feature Selection 
The coding of this technique was done manually and spearately for different combinations of features. We carried out all the experiemnts and stored the results in the "**Grouped_Feature_Selection_All_Results.xlsx**" files. After carrying out all the experiemnts we found out the best group combination and the tested it on the train dataset. The "**Grouped_Feature_Selection_Final_GCEF_Test_Train.py**" contains that final code where we calculated both the train and the test results.

######  Recursive Feature Selection 
In this technique we ranked the features using Random Forest classifier and identified the least important feature and removed it from the train dataset. We ran the loop for 32620 times as we have that many features and chose the feature set with the best accuracy. After choosing the optimal feature set we we tested it on the test dataset.
* "**Recursive_Feature_Selection.py**" contains the entire code for recursive feature selecton.
* "**Recursive_Best_Feature_Set_Train_Test.py**" contains the code where we only ran the code till the optimal set of features was reached and then tested the feature set on the testing data.
