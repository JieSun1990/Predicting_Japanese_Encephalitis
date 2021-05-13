# Machine Learning Algorithms for Predicting the Force of Infection (FOI) of Japanese Encephalitis (JE)
# Summary
This is a follow-up of the previous research done by [Duy M.Nguyen and Quan M.Tran](https://github.com/m2man/JERFOUCRU), with guidance from Dr Hannah Clapham, at the National University of Singapore. 

The project's objective is to predict the FOI of Japanese Encephalitis in various regions. The previous work used Random Forest, while I explored the following methods: 

- PCR
- Random Forest
- Gradient Boosting
- XGBoost

The objective of my work is to provide python code examples of the above methods for future colleagues. We will also compare the methods, explain the hyperparameter tuning and discuss the prediction interval calculation that is less commonly mentioned for machine learning.  

# Data
Data was generated by [Duy M.Nguyen and Quan M.Tran's](https://github.com/m2man/JERFOUCRU) processing steps in Part 1, step 1-7. 

### Input: 
- EM_Imputed_Features_Study.csv
- Imputed_Features_Endemic.csv 
Location: **Generate/Python_CSV/** 

### Data description:
26 X variables:

- The bioclimatic conditions in the word, derived from rainfall and temperature on a monthly basis: 
'Bio_01', 'Bio_02', 'Bio_03', 'Bio_04', 'Bio_05', 'Bio_06', 'Bio_07', 'Bio_08', 'Bio_09', 'Bio_10', 'Bio_11', 'Bio_12', 'Bio_13', 'Bio_14', 'Bio_15', 'Bio_16', 'Bio_17', 'Bio_18', 'Bio_19'
- The Japanese Encephalitis Vector Distribution: 'VD'
- Rice distribution: 'Rice'
- Livestock distribution: 'Pigs'
- Population density: 'DG_000_014bt_dens'
- Specific age group density: 'Adjusted_Pop_Count_2015'
- Urban/Rural category: 'UR'
- Elevation: 'Elv'

1 Y variable: FOI. Note that FOI's range is between 0 to 0.5. You can understand this as a kind of probability, but there is an upper limit to the FOI in this dataset. 

Detailed data description can be found [here](https://github.com/m2man/JERFOUCRU/tree/master/Writing%20Documents).

# Method
I tried 4 methods for this problem. 
- PCR
- Random Forest
- Gradient Boosting
- XGBoost

Please check the respective notebook here: **Notebooks/**


