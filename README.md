Vehicle Emissions Analysis
This project focuses on analyzing vehicle emissions data using various machine learning techniques. The dataset used for analysis contains information about vehicle emissions, including engine size, CO2 emissions, transmission type, smog level, and fuel consumption.

Overview
The main objectives of this project include:
Data loading and preprocessing
Exploratory data analysis
Feature engineering
Building and evaluating machine learning models.

Data Loading and Preprocessing:

Imported necessary libraries such as Pandas and NumPy for data manipulation and analysis.
Loaded the dataset Vehicle_Emissions_Data.csv using Pandas.
Checked basic information about the dataset like size, info, and presence of null values.
Handled missing values by replacing them with mean values.

Exploratory Data Analysis (EDA):
Described the dataset to gain insights into its statistical properties.
Visualized distributions of numeric variables using histograms and box plots.
Explored correlations among numeric variables using correlation matrices.

Feature Engineering:
Created a new categorical variable Engine_Size_Category based on the Engine_Size variable.
Transformed numeric features by normalizing them.

Data Visualization:
Utilized Matplotlib to create various types of plots like histograms, bar plots, and box plots.
Visualized distributions and relationships among variables to understand the data better.

Machine Learning Models:
Implemented linear regression to predict Engine_Size based on CO2_Emissions.
Used logistic regression for classification tasks, predicting Engine_Size_Category.
Employed support vector machine (SVM) for classification tasks as well.

Model Evaluation:
Evaluated the performance of machine learning models using metrics like accuracy score and classification report.

Data Splitting:
Split the dataset into training and testing sets using train_test_split function from sklearn.model_selection.

Linear Regression Model:
Implemented linear regression using the LinearRegression class from sklearn.linear_model.
Utilized the model to predict the Engine_Size based on the CO2_Emissions.
Trained the linear regression model on the training data.
Extracted model coefficients and intercept to understand the relationship between variables.

Data Splitting and Feature Selection for Linear Regression:
Split the dataset into training and testing sets specifically for the linear regression model using train_test_split from sklearn.model_selection.
Segregated predictor variables (X_train, X_test) and the target variable (Y_train, Y_test) to train and evaluate the linear regression model.
Selected relevant features (CO2_Emissions) as predictors for the linear regression model.
Used the selected feature to train the linear regression model for predicting Engine_Size.

Model Evaluation for Linear Regression:
Evaluated the performance of the linear regression model using Mean Squared Error (MSE) metric.
Calculated the MSE to quantify the goodness of fit between the predicted Engine_Size and the actual Engine_Size.




Note
This project was developed using Google Colab, and the notebook PROJECT.ipynb. 
*Uploaded both ipynb format and py format
