# student_intervention
Supervised Machine Learning system evaluating different classification algorithms to implement a system that can predict when a student might be in risk of dropping off, so an intervention can be made in time.

## Project Overview

As education has grown to rely more on technology, vast amounts of data has become available for examination and prediction. Logs of student activities, grades, interactions with teachers and fellow students, and more, are now captured in real time through learning management systems like Canvas and Edmodo. This is especially true for online classrooms, which are becoming popular even at the primary and secondary school level. Within all levels of education, there exists a push to help increase the likelihood of student success, without watering down the education or engaging in behaviors that fail to improve the underlying issues. Graduation rates are often the criteria of choice, and educators seek new ways to predict the success and failure of students early enough to stage effective interventions.


# Description

A local school district has a goal to reach a 95% graduation rate by the end of the decade by identifying students who need intervention before they drop out of school. As a software engineer contacted by the school district, your task is to model the factors that predict how likely a student is to pass their high school final exam, by constructing an intervention system that leverages supervised learning techniques. The board of supervisors has asked that you find the most effective model that uses the least amount of computation costs to save on the budget. You will need to analyze the dataset on students' performance and develop a model that will predict the likelihood that a given student will pass, quantifying whether an intervention is necessary.

# Software Requirements

This project uses the following software and Python libraries:
Python 2.7
NumPy
pandas
scikit-learn (v0.17)
You will also need to have software installed to run and execute a Jupyter Notebook.

# Code
Template code is provided in the student_intervention.ipynb notebook file. You will also be required to use the student-data.csv dataset file to complete your work. 

# Run
In a terminal or command window, navigate to the top-level project directory student_intervention/ (that contains this README) and run one of the following commands:

```bash
ipython notebook student_intervention.ipynb
```
or

```bash
jupyter notebook student_intervention.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

# Project Solution Steps

1. Load the Data into a Pandas DataFrame
2. Explore the data and get some general stats
3. Number of records, number of features, classification output label percentages, etc
4. Prepare data: separate feature columns from target column
5. Prepare data: separate feature data from target data
6. Prepare data: print first few rows (head) to look for non-numeric values
7. Prepara data: replace non-numeric data for numeric data. Features with yes/no values can be replaced for boolean values (1/2). Categorical data can be replaced by adding dummy variables, so all features have boolean values. We typically create a new pandas DataFrame and use get_dummies() pandas method to create the dummy variables for the categorical data.
8. Training and Testing Data Split: split the test and training data. Typically you will split the data randomly, and set apart 25% of the data for testing. We normally use train_test_split from sklearn.model_selection and pass the arrays (features and labels), test size and train size, along with the random state.
9. Choose potential algorithms to use, by evaluating the strengths and weaknesses in the context of the data you have and the problem you are trying to solve. Look into real world application examples, strengths, weaknesses and justification for why it should work in this context.
11. Setup the code to train and test the classifiers. 
    - train_classifier - takes as input a classifier and training data and fits the classifier to the data.
    - predict_labels - takes as input a fit classifier, features, and a target labeling and makes predictions using the F1 score.
    - train_predict - takes as input a classifier, and the training and testing data, and performs train_clasifier and predict_labels. This function will report the F1 score for both the training and testing data separately.
12. Implement the model performance metrics, train and test the performance of the model on different training dataset sizes to evaluable performance as more data is added.
13. Annotate results in terms of training time, test time, training and testing scores and analytics the results across the different dataset sizes. 
14. Choose the best model and justify the response taking into account data samples, features and dimensionality and the analysed behaviour for the unturned model. 
15. Use Grid Search to implement model tuning. Implement performance scoring, and compare the results of the tuned model to the unturned models.
16. Compare the second best algorithm implementing performance tuning.
17. Draw conclusions on analysis and final worlds about the best algorithm fit. 

# Algorithm Research

Gaussian Naive Bayes
Decision Trees
Ensemble Methods (Boosting - AdaBoost)
K-NN
Stochastic Gradient Descent (SGDC)
Support Vector Machines (SVM)
Logistic Regression

# Data

The dataset used in this project is included as student-data.csv. This dataset has the following attributes:

- `school` : student's school (binary: "GP" or "MS")
- `sex` : student's sex (binary: "F" - female or "M" - male)
- `age` : student's age (numeric: from 15 to 22)
- `address` : student's home address type (binary: "U" - urban or "R" - rural)
- `famsize` : family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
- `Pstatus` : parent's cohabitation status (binary: "T" - living together or "A" - apart)
- `Medu` : mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
- `Fedu` : father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
- `Mjob` : mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
- `Fjob` : father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
- `reason` : reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
- `guardian` : student's guardian (nominal: "mother", "father" or "other")
- `traveltime` : home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
- `studytime` : weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
- `failures` : number of past class failures (numeric: n if 1<=n<3, else 4)
- `schoolsup` : extra educational support (binary: yes or no)
- `famsup` : family educational support (binary: yes or no)
- `paid` : extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
- `activities` : extra-curricular activities (binary: yes or no)
- `nursery` : attended nursery school (binary: yes or no)
- `higher` : wants to take higher education (binary: yes or no)
- `internet` : Internet access at home (binary: yes or no)
- `romantic` : with a romantic relationship (binary: yes or no)
- `famrel` : quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
- `freetime` : free time after school (numeric: from 1 - very low to 5 - very high)
- `goout` : going out with friends (numeric: from 1 - very low to 5 - very high)
- `Dalc` : workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
- `Walc` : weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
- `health` : current health status (numeric: from 1 - very bad to 5 - very good)
- `absences` : number of school absences (numeric: from 0 to 93)
- `passed` : did the student pass the final exam (binary: yes or no)


