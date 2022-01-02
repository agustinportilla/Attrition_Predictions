# Attrition_Predictions

**Project Domain**: Human Resources

**Tools used**: Python

**Type of Algorithms used**: Logistics Regression, RandomTree & RandomForest, Deep Learning.

**Project Summary**:

The objective of this work is to compare the performance of different algorithms to predict our target variable: Attition. In the area of Human Resources, Attrition is defined as when one employee decides to leave the company (without being fired).

**Details about the Dataset**:

The Dataset Human_Resources.csv contains 1470 records of different employees. It contains the following variables:

   - **Age**; **Attrition**; **BusinessTravel**; **DailyRate**; **Department**; **DistanceFromHome**; **Education**; **EducationField**; **EmployeeCount**; **EmployeeNumber**; - **EnvironmentSatisfaction**; **Gender**; **HourlyRate**; **JobInvolvement**; **JobLevel**; **JobRole**; **JobSatisfaction**; **MaritalStatus**; **MonthlyIncome**; **MonthlyRate**; **NumCompaniesWorked**; **Over18**; **OverTime**; **PercentSalaryHike**; **PerformanceRating**; **RelationshipSatisfaction**; **StandardHours**; **StockOptionLevel**; **TotalWorkingYears**; **TrainingTimesLastYear**; **WorkLifeBalance**; **YearsAtCompany**; **YearsInCurrentRole**; **YearsSinceLastPromotion**; **YearsWithCurrManager**

**Dataset Preparation**:

  - **Libraries used**: pandas, numpy, seaborn, matplotlib.pyplot, sklearn.preprocessing (OneHotEncoder, MinMaxScaler and train_test_split) and sklearn.model_selection (train_test_split).
  
  - **Initial Data Preparation**: 
      - We used **describe** to spot any missing numeric field; 
      - We converted some categorical fields (**Attrition**, **OverTime** and **Over18**) into numerical, using **replace**;
      - We created multiple **histograms**. This helped us to identify some columns that we will not need. As a result, we dropped the columns **EmployeeCount**, **StandardHours**,**EmployeeNumber**
      - We explored our target variable (**Attrition**). We created two new datasets using this variable and we compared them. We disconvered some strong correlations between our target variable and others, for example **Age** (young people tend to leave the company more often) and **DistanceFromHome** (Employees that live further away have more chances of leaving).

  
  - **Data Visualization**: We applied many visualization techniques to better understand our dataset.
      - A **Heatmap** showed us some more strong correlation between differet variables.
      - We used **Countplots** and **Kdeplots** to further explore the relation between our target variable and others.
      - Finally, we used **Boxplots** to visualize the distribution of salary among different Job Roles.
      
  - **Final Data Preparation**: 
      - We used **OneHotEncoder** to create dummy columns for our categorical variables.
      - We used **MinMaxScaler** to normalize our data.
      - We used **train_test_split** to create training and testing datasets.

**LogisticRegression**

  - **Libraries used**: sklearn.linear_model (LogisticRegression) and sklearn.metrics (accuracy_score)
      
  - We created our Model, we fitted it and we obtained the following results:
    - Accuracy 87.22826086956522 %
    - Precision 85.71428571428571 %
    - Recall 29.03225806451613 %
    - F1Score 43.373493975903614 %

  - We tried using different Threshold. We discovered that using a Threshold of 0.3 improves the overall performance:
    - Accuracy 86.1413043478261 %
    - Precision 58.730158730158735 %
    - Recall 59.67741935483871 %
    - F1Score 59.199999999999996 %

  - This improvement means that our Accuracy remains the same while our F1Score improves significantly. At the same time, our Recall improves by 30%. However, our Precision decreases by 27%. If we are in a business context in which we need to reduce False Negatives, this can be considered a big improvement.

**Random Forest Model**

  - **Libraries used**: sklearn.ensemble (RandomForestClassifier)

  - We created our Model, we fitted it and we obtained the following results:
    - Accuracy 84.51086956521739 %
    - Precision 77.77777777777779 %
    - Recall 11.29032258064516 %
    - F1Score 19.718309859154928 %

  - We ran the Model multiple times, using different quantities of trees (n_estimators): 500, 1000, 2500, 5000, 10000, 15000, 20000.
  - We found out that from 500 to 2500 trees the F1Score keeps changing. But it remains the same with 5000, 10000, 15000 and 20000 tress.
  - As a result, the model can not be improved by adding extra trees.

**Random Tree - Cross Validation Model**

  - All the models that we have used so far, along with the Deep Learning Model (that will be used next) require a training and a testing dataset to work. The Random Tree Cross Validation model works different. It uses our dataset multiple times. With each iteration, the train_test split is run again. This means that every single value is normally used for testing and for training. In practise, this is a very big advantage when working with small datasets.

  - **Libraries used**: sklearn.model_selection (KFold and cross_val_score) and sklearn.tree (DecisionTreeClassifier)

  - We created our Model, we fitted it and we obtained the following results:
    - Accuracy 85.85034013605443 %
    - Precision 67.05882352941175 %
    - Recall 24.050632911392405 %
    - F1Score 35.40372670807453 %

**Deep Learning Model**

- **Libraries used**: tensorflow

- We created our Model, we fitted it and we obtained the following results:
    - Accuracy 83.69565217391305 %
    - Precision 52.63157894736842 %
    - Recall 32.25806451612903 %
    - F1Score 40.0 %

![image](https://user-images.githubusercontent.com/89322259/147881109-292662cf-942e-42a6-b85a-603882d3a62b.png)


**Which model is better?**

  - Picking the best model will depend on our objective. For this particular case, our Deep Learning Model does not present any benefit (based on the selected KPI's). The two models that perform better are Logistic Regression and Logistic Regression (Threshold:3).
  - If we want to reduce the number of False Negative we have, we need to increase our Recall. In that case, we would benefit from the Logistic Regression (Threshold:3) model.
  - If we want to reduce the number of False Positive, we need to increase our Precision. In that case, we would benefit from the Logistic Regression model.

**How can we improve our model**

  - If we want to improve the performance of our model, it would be a good practise to increase the number of records in our Dataset. Currently, our testing dataset consist of 368 records. With such a low number, the representativeness of the sample is comprised. As a result, if we would run our train_test_split multiple times, we would get different results for all our models (except for CV).


