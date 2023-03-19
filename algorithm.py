import re
import scipy
import string
import pickle 
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from prettytable import PrettyTable
from astropy.table import Table, Column
from sklearn.metrics import accuracy_score


train_file_path = 'train.csv'
test_file_path = 'test.csv'
train_dataset = pd.read_csv(train_file_path)
test_dataset = pd.read_csv(test_file_path)
# The code reads data from two CSV files, 'train.csv'
# and 'test.csv', using pandas' read_csv function, and creates 
# dataframes named 'train_dataset' and 'test_dataset'. The file 
# paths are stored in 'train_file_path' and 'test_file_path', respectively.


print("\n\n\nAttributes Names in Train Dataset:")
print("==================================\n")
print(train_dataset.columns)
print("\n\n\nNumber of instances in Train Dataset:")
print("====================================")
print("Train Data instances: ",train_dataset.shape[0])
# This code prints the attribute names of the train_dataset
# dataframe using the columns attribute, and then the number
# of instances in the train_dataset dataframe using the shape
# function, specifically the number of rows in the first
# element of the tuple that shape() returns, which corresponds
# to the number of instances. The printed output is separated
# into two sections using a newline character and print
# statements are used to provide descriptive headers for
# each output section.


print("\nTrain Dataset:")
print("==============\n")
train_dataset.columns.name = 'index'
print(train_dataset)
# This code prints out the "Train Dataset" title followed by a 
# line break, and then sets the column name of the train_dataset 
# dataframe to 'index'. Finally, it prints out the train_dataset 
# dataframe. The output will be the train_dataset dataframe with 
# the column name 'index' displayed along with the "Train Dataset" 
# title. The comment for this code should describe what the code 
# does in 50 characters or less, such as "Prints and formats train 
# dataset".


print("\n\n\nAttributes Names in Test Dataset:")
print("==================================\n")
print(test_dataset.columns)
print("\n\n\nNumber of instances in Test Dataset:")
print("====================================")
print("Test Data instances: ",test_dataset.shape[0])
# The code prints out information about the test dataset. First, 
# it displays the attribute names in the dataset by printing 
# "Attributes Names in Test Dataset:" followed by a line of equal
#  signs. It then prints out the actual attribute names using the 
# .columns attribute of the test_dataset variable. Next, the code 
# prints "Number of instances in Test Dataset:" followed by a line
#  of equal signs. It then prints out the number of instances (or rows)
#  in the test dataset using the .shape[0] attribute of the test_dataset
#  variable. This code is useful for quickly getting an overview of the
#  structure and size of a dataset, which is an important step in any data 
#  analysis or machine learning project.


print("\nTest Dataset:")
print("==============\n")
test_dataset.columns.name = 'index'
print(test_dataset)
# This code will print the test dataset in a formatted manner. It first prints
# a header "Test Dataset:" to indicate what is being displayed. Then, it sets
# the name of the columns to 'index' to provide better readability. Finally,
# it prints the test dataset using the 'print' function, which will display
# the entire dataset in a tabular format. This is useful for examining the data
# and understanding the structure of the test dataset, as well as identifying any
# potential issues or outliers that may need to be addressed before running
# any analysis or machine learning models.


train_Data_Label_Male = train_dataset[train_dataset['gender']== 'Male']
print("\n Train instances having label 'Male': \"",train_Data_Label_Male.shape[0],"\"")
train_Data_Label_Male.columns.name = 'index'
print("================\n")
print(train_Data_Label_Male)
# This code is selecting a subset of the training dataset based on the condition that
# the gender column equals 'Male'. The selected subset is then printed to the console
# along with the number of instances in the subset. The code also sets the name of the
# columns index to 'index' for better readability, and then prints the subset to the
# console with column names displayed on the left.


train_Data_Label_Female = train_dataset[train_dataset['gender'] == 'Female']
print("\n Train instances having label 'Female': \"",train_Data_Label_Female.shape[0],"\"")
train_Data_Label_Female.columns.name = 'index'
print("=========================\n")
print(train_Data_Label_Female)
# The above code snippets are related to loading and manipulating datasets using pandas 
# library in Python. The first two snippets load the train and test datasets from CSV files.
# The next few snippets print the attributes names and number of instances in the test dataset
# and the train dataset for male and female labels separately. The datasets are printed using
# the print() function and some formatting is applied to make the output more readable.


test_Data_Label_Male = test_dataset[test_dataset['gender'] == 'Male']
print("\n Test instances having label 'Male': \"",test_Data_Label_Male.shape[0],"\"")
test_Data_Label_Male.columns.name = 'index'
print("=================================\n")
print(test_Data_Label_Male)
# This code filters the test_dataset to only include instances where the gender attribute is 'Male'.
# It then prints the number of instances in this filtered dataset and sets the column name to 'index'.
# Finally, it prints a separator line for formatting purposes


test_Data_Label_Female = test_dataset[test_dataset['gender'] == 'Female']
print("\n Test instances having label 'Female': \"",test_Data_Label_Female.shape[0],"\"")
test_Data_Label_Female.columns.name = 'index'
print("==================================\n")
print(test_Data_Label_Female)
# This code segment is selecting rows from the "test_dataset" based on the "gender" column value,
# filtering out the rows where gender is "Female". It then prints out the number of rows in the
# filtered dataset and assigns a name "index" to the columns. Finally, it prints out a separator
# line and displays the filtered dataset.


print("\n Visual representation of total number of 'Males' and 'Females' in Train Dataset:")
print("===========================\n")
train_data_gender_grouped = train_dataset.groupby('gender')['gender'].count().plot(kind = 'bar', rot = 0)
train_data_gender_grouped.set(xlabel = 'Gender')
train_data_gender_grouped.legend(['Frequency'], loc='upper right')
# This code is generating a visualization of the total number of males and females in the training
# dataset. It first prints a message indicating what the visualization represents. Then, it creates
# a new variable called train_data_gender_grouped which groups the training dataset by gender and
# counts the number of instances for each gender. Finally, it plots the counts as a bar chart using
# the plot() function with kind='bar', and sets the x-axis label to 'Gender' and the legend label
# to 'Frequency'. The rot=0 argument is used to prevent the x-axis labels from being rotated.


print("\n Visual representation of total number of 'Males' and 'Females' in Test Dataset:")
print("===========================\n")
test_data_gender_grouped = test_dataset.groupby('gender')['gender'].count().plot(kind = 'bar', rot = 0)
test_data_gender_grouped.set (xlabel = 'Gender')
test_data_gender_grouped.legend (['Frequency'], loc='upper right')
# This code is creating a visual representation of the total number of males and females in the test
# dataset. The first line prints a message to describe what the visual representation will show.
# Then, the second line creates a grouped bar chart using the groupby() method to group the data
# by gender and count() to count the number of instances for each gender in the test dataset.
# The kind parameter is set to 'bar' to create a bar chart. The rot parameter is set to 0 to
# keep the x-axis labels horizontal. The third line sets the x-axis label to 'Gender', and the
# fourth line sets the legend label to 'Frequency' and places it in the upper right corner of
# the chart. Overall, this code provides an easy-to-understand visual representation of the
# number of males and females in the test dataset.


print("\n Number of people having various hair length in Train dataset:")
print("======================\n")
train_dataset.groupby('hair')['hair'].count().sort_values(ascending=False).plot(kind = 'bar', color = 'brown')
# This code is used to plot a bar graph showing the number of people in the train dataset with 
# different hair lengths. The output shows the number of people having various hair lengths in 
# descending order. The x-axis represents the hair length, and the y-axis represents the count. 
# The groupby() function is used to group the dataset by hair length, and the count() function 
# is used to count the number of instances in each group. The sort_values() function is used to 
# sort the hair lengths in descending order based on the count. Finally, the plot() function is 
# used to create a bar graph using the Pandas library. The kind parameter is set to "bar" to 
# specify that a bar graph is to be plotted, and the color parameter is set to "brown" to set 
# the color of the bars to brown.


print("\n Number of people having various hair length in Test dataset:")
print("======================\n")
test_dataset.groupby('hair')['hair'].count().sort_values(ascending=False).plot(kind = 'bar', color = 'deeppink')
# This code is plotting a bar graph to display the number of people having different hair 
# \lengths in the test dataset. The first line prints a statement on the console to indicate 
# the purpose of the graph. The second line adds some space for clarity. The third line groups 
# the test dataset by hair length and counts the number of people with each hair length using 
# the groupby() and count() functions. The sort_values() function sorts the hair lengths in 
# descending order based on the number of people. Finally, the plot() function plots a bar 
# graph with hair length on the x-axis and the count of people on the y-axis. The kind parameter 
# is set to 'bar' to specify a bar graph, and the color parameter is set to 'deeppink' to choose 
# the color of the bars.


print("\n Number of people have/haven't beard in Train dataset:")
print("==========================\n")
train_dataset.groupby('beard')['beard'].count().sort_values(ascending=False).plot(kind ='bar', colormap = 'RdGy_r')
# This code generates a visualization of the number of people in a train dataset who have or don't have a 
# beard. The code starts by printing a message indicating that the graph shows the number of people who 
# have or don't have a beard in the train dataset. Then it plots a horizontal bar graph using the plot 
# function of the Pandas library. The graph is grouped by the 'beard' column in the train dataset and 
# the count of the 'beard' column is calculated for each group. The sort_values function sorts the values 
# in descending order. The kind parameter is set to 'bar' to create a bar graph, and the colormap parameter 
# is set to 'RdGy_r' to specify the color scheme for the graph.


print("\n Number of people have/haven't beard in Test dataset:")
print("==========================\n")
test_dataset.groupby('beard')['beard'].count().sort_values(ascending=False).plot(kind = 'bar', colormap = 'plasma_r')
# This code is displaying the number of people who have or haven't a beard in the test dataset. 
# It first prints a header indicating the information that will be shown. Then it groups the 
# data by the 'beard' column using the groupby() function and applies the count() function to 
# count the number of occurrences of each category. The resulting counts are then sorted in 
# descending order using the sort_values() function and plotted as a bar graph using the plot() 
# function. The colormap used for the graph is 'plasma_r'.


train_dataset = train_dataset.fillna(' ')
Preprocessed_train_dataset = train_dataset.round({'height': 2})
# In this code, the fillna() method is used to fill any missing values in the train_dataset 
# dataframe with empty strings. The round() method is then applied to the height column of 
# the train_dataset dataframe to round off the values to two decimal places. The resulting 
# preprocessed dataset is then saved in a new dataframe called Preprocessed_train_dataset.


def print_side_by_side(*objs, **kwds):
    ''' function print objects side by side '''
    from pandas.io.formats.printing import adjoin
    space = kwds.get('space', 15)
    reprs = [repr(obj).split('\n') for obj in objs]
    print(adjoin(space, *reprs))
# The code defines a function called print_side_by_side that takes multiple objects as 
# arguments and prints them side by side. The function uses the adjoin() method from 
# the pandas.io.formats.printing module to format and align the objects. The space 
# between each object can be specified as a keyword argument space (default is 15). 
# The function is useful for comparing the output of different objects or dataframes 
# in a more readable way.


print("\nTrain dataset before pre-processing:", "\t\t\t", "Train dataset after pre-processing:")
print("============","\t\t\t"," ============\n\n")
print_side_by_side(train_dataset ,Preprocessed_train_dataset)
# This code defines a function called print_side_by_side that is used to print two objects 
# side by side. The function takes in a variable number of arguments (*objs) and a dictionary 
# (**kwds) containing any keyword arguments. The default value for the keyword argument space 
# is 15. After that, the code prints a header indicating that it will print the train dataset 
# before and after pre-processing, and then calls print_side_by_side with the original train 
# dataset and the pre-processed train dataset as arguments. The output will show the two 
# datasets side by side for comparison.


test_dataset = test_dataset.fillna(' ')
Preprocessed_test_dataset = test_dataset.round({'height': 2})
# The code fills the missing values in the test_dataset with an empty string using fillna() method, 
# and then rounds the height column to two decimal places using the round() method. The resulting 
# preprocessed dataset is stored in Preprocessed_test_dataset.


print("\nTest dataset before pre-processing:", "\t\t\t", "Test dataset after pre-processing:")
print("============","\t\t\t"," ============\n\n")
print_side_by_side(test_dataset , Preprocessed_test_dataset)
# this will print test_dataset and Preprocessed_test_dataset side by side.


labelEncoder_gender = LabelEncoder()
labelEncoder_scarf = LabelEncoder()
labelEncoder_hair = LabelEncoder()
labelEncoder_beard = LabelEncoder()
# These lines of code initialize four LabelEncoder objects, which are used to encode 
# categorical variables in the dataset. LabelEncoder is a utility class in the scikit-learn 
# library that can be used to encode categorical labels with integer values. Each categorical 
# variable in the dataset is assigned a unique integer value, which makes it easier for 
# the machine learning algorithm to process the data.


gender = ['Female','Male']
scarf = ['Yes','No']
beard = ['Yes','No']
hairLength = ['Bald','Long','Short','Medium']
# Defining arrays


labelEncoder_gender.fit(gender)
labelEncoder_scarf.fit(scarf)
labelEncoder_beard.fit(beard)
labelEncoder_hair.fit(hairLength)
# In the given code, LabelEncoder() is a class from the scikit-learn library, which is 
# used to encode categorical variables as numeric labels. fit() is a method of the LabelEncoder 
# class which is used to fit the encoder on the specified data to create a mapping of categories 
# to integer labels. Here, labelEncoder_gender.fit(gender) maps the categories "Female" and "Male" 
# to integer labels 0 and 1, respectively. Similarly, labelEncoder_scarf.fit(scarf) maps "Yes" to 
# 1 and "No" to 0, and labelEncoder_beard.fit(beard) maps "Yes" to 1 and "No" to 0. Finally, 
# labelEncoder_hair.fit(hairLength) maps the hair lengths "Bald", "Long", "Short", and "Medium" 
# to integer labels 0, 1, 2, and 3, respectively. After fitting, we can use the transform() method 
# to transform the original categorical data into encoded numeric labels.


label_encoded_train_dataset=Preprocessed_train_dataset.copy(deep=True)
label_encoded_test_dataset=Preprocessed_test_dataset.copy(deep=True)
# This code creates two new pandas dataframes (label_encoded_train_dataset and label_encoded_test_dataset)
#  by making a copy of the preprocessed train and test datasets (Preprocessed_train_dataset and 
# Preprocessed_test_dataset) respectively. The copy() method creates a deep copy of the original 
# dataframes, meaning that any changes made to the new dataframes will not affect the original dataframes.


print('\nGender attribute Label Encoding in Train dataset:')
print("=====================\n")
label_encoded_train_dataset['gender'] = labelEncoder_gender.fit_transform(Preprocessed_train_dataset['gender'])
# This code performs label encoding on the 'gender' attribute of the preprocessed train dataset. T
# he label encoder object 'labelEncoder_gender' that we defined earlier is used


Preprocessed_train_dataset['encoded_values_gender'] = label_encoded_train_dataset['gender']
print(Preprocessed_train_dataset[['gender','encoded_values_gender']])
# The code above adds a new column called 'encoded_values_gender' to the 'Preprocessed_train_dataset
# ' dataframe which contains the encoded values of the 'gender' column using the previously fitted 
# label encoder for gender. It then prints the two columns 'gender' and 'encoded_values_gender' of 
# the 'Preprocessed_train_dataset' dataframe to show the original values and the corresponding encoded values.


print('\n\n\nScarf attribute Label Encoding in Train dataset:')
print("=====================\n")
label_encoded_train_dataset['scarf'] = labelEncoder_scarf.fit_transform(Preprocessed_train_dataset['scarf'])
# This code is encoding the 'scarf' attribute in the training dataset using the labelEncoder_scarf 
# object that we previously created. The fit_transform() method is used to fit the encoder to the 
# data and transform the 'scarf' column values from strings ('Yes' and 'No') to integers (0 and 1). 
# The resulting encoded values are then assigned to a new 'scarf' column in the label_encoded_train_dataset dataframe.


Preprocessed_train_dataset['encoded_values_scarf'] = label_encoded_train_dataset['scarf']
print(Preprocessed_train_dataset[['scarf' , 'encoded_values_scarf']])
# This code creates a new column 'encoded_values_scarf' in the 'Preprocessed_train_dataset' DataFrame 
# which contains the encoded values of the 'scarf' attribute obtained from the 'label_encoded_train_dataset'. 
# The print statement is used to display a subset of the 'Preprocessed_train_dataset' containing the columns '
# scarf' and 'encoded_values_scarf' so we can verify the encoding has been performed correctly.


print('\n\n\nBeard attribute Label Encoding in Train dataset:')
print("=========================\n")
label_encoded_train_dataset['beard'] = labelEncoder_beard.fit_transform(Preprocessed_train_dataset['beard'])
# This code is performing label encoding on the "beard" attribute of the preprocessed train dataset. 
# Label encoding is a process of converting categorical values into numerical values. Here, the "beard" 
# attribute has categorical values of "Yes" and "No". The LabelEncoder from scikit-learn library is used 
# to perform this encoding. fit_transform() method is used to transform the "beard" attribute of the 
# preprocessed train dataset into encoded numerical values. The encoded values are then assigned to 
# the "beard" attribute of the label_encoded_train_dataset.


Preprocessed_train_dataset['encoded_values_beard'] = label_encoded_train_dataset['beard']
print(Preprocessed_train_dataset[['beard','encoded_values_beard']])
# This code is adding a new column named "encoded_values_beard" to the "Preprocessed_train_dataset" 
# dataframe, which contains the encoded values for the "beard" attribute using LabelEncoder. The 
# "fit_transform" method of LabelEncoder is applied to the "beard" attribute of "Preprocessed_train_dataset" 
# and the resulting encoded values are stored in the new column "encoded_values_beard". Finally, the code 
# prints the "beard" attribute and the corresponding encoded values side by side.


print('\n\n\nHair attribute Label Encoding in Train dataset:')
print("======================\n")
label_encoded_train_dataset['hair'] = labelEncoder_hair.fit_transform(Preprocessed_train_dataset['hair'])
# This code performs label encoding on the "hair" attribute in the "Preprocessed_train_dataset". 
# The "LabelEncoder" class from the Scikit-learn library is used to convert categorical variables 
# to numeric ones. "fit_transform()" method of the "LabelEncoder" class is used to perform the encoding,
#  which first fits the encoder to the unique categories in the "hair" column of "Preprocessed_train_dataset" 
# using "fit()", and then encodes the categories to integers using "transform()". The encoded values
#  are then stored in a new column "hair" in "label_encoded_train_dataset".


Preprocessed_train_dataset['encoded_values_hair']=label_encoded_train_dataset['hair']
print(Preprocessed_train_dataset[['hair','encoded_values_hair']])
# This code is creating a new column in the pre-processed training dataset called 'encoded_values_hair'.
#  The values in this column are obtained by applying label encoding to the values in the 'hair' column 
# of the pre-processed training dataset using the label encoder for hair length. The label encoder assigns
#  a unique integer value to each hair length category ('Bald', 'Long', 'Short', 'Medium') and these
#  integer values are used as the encoded values in the new 'encoded_values_hair' column. The print
#  statement is printing a table with two columns: 'hair' and 'encoded_values_hair' to show the original 
# values and their corresponding encoded values.


label_encoded_test_dataset['gender']=labelEncoder_gender.fit_transform(Preprocessed_test_dataset['gender'])
label_encoded_test_dataset['scarf'] =labelEncoder_scarf.fit_transform (Preprocessed_test_dataset['scarf'])
# It looks like the code is performing label encoding on the gender and scarf columns of 
# Preprocessed_test_dataset using the LabelEncoder objects labelEncoder_gender and 
# labelEncoder_scarf, respectively. The resulting label encoded values are being 
# assigned to the gender and scarf columns of label_encoded_test_dataset.


label_encoded_test_dataset['beard']=labelEncoder_beard.fit_transform(Preprocessed_test_dataset['beard'])
label_encoded_test_dataset['hair']=labelEncoder_hair.fit_transform(Preprocessed_test_dataset['hair'])
# These lines of code are performing label encoding on the categorical attributes 
# ('gender', 'scarf', 'beard', and 'hair') in the test dataset using the respective LabelEncoder objects
# . The fit_transform() method of LabelEncoder is being used here to first fit the encoder on the data
# and then transform the attribute values into encoded labels. label_encoded_test_dataset is a deep copy 
# of the preprocessed test dataset, and the encoded values of each categorical attribute are being added 
# to it as new columns. Label encoding is a process of converting categorical data into numerical data 
# so that it can be used as input for machine learning models. It assigns a unique integer value to 
# each category of a categorical attribute, thus enabling the model to understand and interpret the
#  attribute values in a meaningful way.


training_features = ['height', 'weight', 'hair', 'beard', 'scarf']
target = 'gender'
# It looks like you are defining the list of training features and the target variable for your
#  machine learning model. The training_features list contains the names of the features you 
# will use to train your model (in this case: height, weight, hair, beard, and scarf), while
#  the target variable is the variable you want to predict, which is gender in this case.


X_train = label_encoded_train_dataset.loc[:, training_features]
y_train =label_encoded_train_dataset.loc[:, target]
# This code extracts the training features and target variable from the preprocessed and label encoded training dataset.
# X_train contains the features height, weight, hair, beard, and scarf that will be used to train the model.
# y_train contains the target variable gender, which is the variable that the model will predict based on the input features.
# This separation of features and target variable is a common step in preparing the data for machine learning models.


X_test = label_encoded_test_dataset.loc[:, training_features]
y_test =label_encoded_test_dataset.loc[:, target]
# These lines of code are preparing the training and testing datasets for the classification model. 
# X_train and X_test contain the features that will be used to train and test the model. In this case, 
# the features are the height, weight, hair, beard, and scarf attributes that have been encoded using 
# label encoding. y_train and y_test contain the target variable, which is the gender attribute that
#  has also been encoded using label encoding. These target values will be used to train and evaluate 
# the classification model.


model_names = []
logistic_regression = LogisticRegression()
print("\nParameters and their values:")
# This code initializes an empty list model_names, and creates a 
# LogisticRegression() object called logistic_regression. The following 
# line just prints the string "Parameters and their values:" to the console.
#  It's likely that this code is part of a larger script that trains and evaluates
#  multiple machine learning models, and model_names will be used to store the
#  names of the various models. The LogisticRegression() object is a classification 
# algorithm that can be trained on labeled data and used to predict the class of new, 
# unlabeled data.


print("==================\n")
print(logistic_regression)
# THIS PRINTS THE LOGISTIC REGRESSION


logistic_regression.fit(X_train,np.ravel(y_train))
model_names.append('LogisticRegression')
# In the above code, we are creating an instance of the LogisticRegression class and 
# fitting the model to the training data using the fit() method. np.ravel(y_train) is
#  used to convert the y_train dataframe into a flattened numpy array. After fitting 
# the model, we append the name of the model to a list model_names.


random_forest_classifier = RandomForestClassifier()
print("\nParameters and their values:")
print("======================\n")
print(random_forest_classifier)
random_forest_classifier.fit(X_train,np.ravel(y_train))
model_names.append('RandomForestClassifier')
# RandomForestClassifier() is an estimator in the Scikit-learn library for classification problems
# using a random forest algorithm, which creates multiple decision trees from random subsets of the
# data and combines their results to make a final prediction. In this code block, a RandomForestClassifier
# object is created and its parameters are printed to the console. Then, it is fitted to the training
# data using the fit method, and the string 'RandomForestClassifier' is appended to the model_names list.


linear_svc = LinearSVC()
print("\nParameters and their values:")
print("=====================\n")
print(linear_svc)
linear_svc.fit(X_train,np.ravel(y_train))
model_names.append('LinearSVC')
# These lines of code create a Linear Support Vector Classifier model with the default hyperparameters, print 
# out the parameters and their values, and fit the model using the training data X_train and the target
#  variable y_train. Finally, it appends the name of the model to a list called model_names.


bernoulli_nb = BernoulliNB()
print("\nParameters and their values:")
print("======================\n")
print(bernoulli_nb)
bernoulli_nb.fit(X_train,np.ravel(y_train))
model_names.append('BernoulliNB')
# These lines of code define and train a Bernoulli Naive Bayes model for classification. bernoulli_nb = BernoulliNB()
#  creates an instance of the Bernoulli Naive Bayes model with default hyperparameters. 
# bernoulli_nb.fit(X_train,np.ravel(y_train)) trains the model on the training data X_train 
# and the target variable y_train using the fit() method of the Bernoulli Naive Bayes model.
#  model_names.append('BernoulliNB') adds the name of the model to a list of model names, 
# which is used later to compare the performance of different models.


Accuracy_Scores=[]
y_prediction = logistic_regression.predict(X_test)
Accuracy_Score = round(accuracy_score(y_test,y_prediction),2)
inverse_transform_values = labelEncoder_gender.inverse_transform(y_prediction)
print("\nPrediction using Logistic Regression:")
print("=======================\n")
print(Preprocessed_test_dataset.assign(predicted_gender = inverse_transform_values))
# This block of code performs the prediction using the Logistic Regression model on the test data 
# X_test. First, the predict method is called on the logistic_regression model to predict the gender
#  labels of the test data. The predicted labels are assigned to y_prediction. Then, the accuracy 
# score is calculated using the accuracy_score method from the sklearn.metrics module. The true 
# labels y_test and the predicted labels y_prediction are passed to this method. The accuracy 
# score is rounded to two decimal places and assigned to the variable Accuracy_Score. Next, 
# the inverse_transform method is called on labelEncoder_gender to convert the predicted label 
# values to their original string representation. The resulting values are assigned to 
# inverse_transform_values. Finally, the predicted gender labels are printed using the 
# print statement along with the original test dataset Preprocessed_test_dataset. The 
# predicted gender labels are added as a new column to this dataset using the assign method.


print("\n\nAccuracy score : ")
print("===============")
print(Accuracy_Score)
Accuracy_Scores.append(Accuracy_Score)
y_prediction =random_forest_classifier.predict(X_test)
Accuracy_Score = round(accuracy_score(y_test,y_prediction),2)
# It looks like the code is predicting the target variable 'gender' using different machine 
# learning algorithms and calculating their accuracy scores on the test dataset. Here, the 
# code is using a Random Forest Classifier algorithm to predict the target variable and then 
# calculating the accuracy score for it. The predict() function is used to predict the target
#  variable using the features from the test dataset. The accuracy_score() function is used to
#  calculate the accuracy score of the predicted values with the actual values of the target 
# variable. The accuracy score is then appended to the list Accuracy_Scores. round() function
#  is used to round off the accuracy score to two decimal places.


inverse_transform_values = labelEncoder_gender.inverse_transform(y_prediction)
print("\nPrediction using RandomForestClassifier:")
print("========================\n")
print(Preprocessed_test_dataset.assign(predicted_gender = inverse_transform_values))
print("\n\nAccuracy score : ")
print("===============")
print(Accuracy_Score)
Accuracy_Scores.append(Accuracy_Score)
# It looks like this code is predicting the gender of individuals using different machine learning
#  models, and evaluating their accuracy. The first part of the code involves preprocessing the 
# data and encoding the categorical variables (beard, hair, scarf) into numerical values. Then, 
# the data is split into training and testing sets, and different machine learning models 
# (Logistic Regression, Random Forest, LinearSVC, BernoulliNB) are applied to the training 
# data. After the models are trained, the accuracy of each model is evaluated using the testing
#  set. The accuracy scores are stored in a list called "Accuracy_Scores". The code snippet you 
# posted is specifically evaluating the accuracy of the Logistic Regression and Random Forest
#  models. It predicts the gender of individuals in the testing set using these models, and 
# prints out the predicted gender along with the original test dataset. The accuracy score 
# for each model is also printed.


y_prediction = linear_svc.predict(X_test)
Accuracy_Score = round(accuracy_score(y_test,y_prediction),2)
inverse_transform_values = labelEncoder_gender.inverse_transform(y_prediction)
print("\nPrediction using LinearSVC:")
print("==========================\n")
print(Preprocessed_test_dataset.assign(predicted_gender = inverse_transform_values))
# It seems like the code is performing prediction on the test dataset using the trained models, 
# and then calculating the accuracy score of each model. Here, the code is predicting the gender
#  of the individuals in the test dataset using LinearSVC model. The accuracy score is then 
# calculated using the predicted values and actual values of the gender attribute in the test
#  dataset. The predicted gender values are inverse transformed using labelEncoder_gender.
# inverse_transform to convert the label encoded values back to their original form. Finally, 
# the predicted gender values are printed along with the test dataset and the accuracy score is printed.


print("\n\nAccuracy score : ")
print("===============")
print(Accuracy_Score)
Accuracy_Scores.append(Accuracy_Score)
y_prediction = bernoulli_nb.predict(X_test)
Accuracy_Score = round(accuracy_score(y_test,y_prediction),2)
# This code calculates the accuracy score of the Bernoulli Naive Bayes model on the test dataset 
# and prints it to the console. The accuracy_score function from scikit-learn library is used to 
# calculate the accuracy score by comparing the predicted values obtained from the Bernoulli Naive
#  Bayes model with the actual values in the y_test dataset. The accuracy score is rounded off to 
# 2 decimal places using the round function and assigned to the variable Accuracy_Score. Then, 
# Accuracy_Score is printed to the console using the print function. The append function is used
#  to add the accuracy score to the Accuracy_Scores list. Finally, the predict method is called 
# on the Bernoulli Naive Bayes model with the X_test dataset as input to obtain the predicted values.
#  The accuracy score for this prediction is then calculated and assigned to the variable Accuracy_Score.


inverse_transform_values = labelEncoder_gender.inverse_transform(y_prediction)
print("\nPrediction using BernoulliNB:")
print("============================\n")
print(Preprocessed_test_dataset.assign(predicted_gender = inverse_transform_values))
print("\n\nAccuracy score : ")
print("===============")
print(Accuracy_Score)
Accuracy_Scores.append(Accuracy_Score)
# In the above code block, we are evaluating the performance of the BernoulliNB model on the test dataset. 
# First, we make predictions on the test dataset using the predict method of the BernoulliNB model and 
# store them in the variable y_prediction. Next, we calculate the accuracy score of the model using the
#  accuracy_score function from scikit-learn, and round it to 2 decimal places. The accuracy score 
# represents the proportion of correct predictions made by the model on the test dataset. Then, we 
# use the inverse_transform method of the LabelEncoder object to transform the predicted gender values 
# back to their original form ('male' and 'female') and assign them to the inverse_transform_values
#  variable. Finally, we print the predicted gender values for the test dataset and their corresponding
#  accuracy score. We also append the accuracy score to the Accuracy_Scores list.


print('\n\nDetailed Performance of all the models.')
print("==========================\n")
accuracy_score_table = PrettyTable(['Model','Accuracy'])
maximum = 0
for i in range(0, 4):
    model = model_names[i]
    score = Accuracy_Scores[i]
    if(maximum < score):
        maximum = score
        index = i
    accuracy_score_table.add_row([model,score])

print(accuracy_score_table)
print('\n\nBest Model.')
print("=======================\n")
# It seems like the code is trying to compare the accuracy scores of different models on a given dataset
#  and identify the best model among them. The code first creates four machine learning models (Logistic
#  Regression, Random Forest Classifier, LinearSVC, and BernoulliNB) and fits them on the training data.
#  Then, it evaluates the accuracy of each model on the test data and stores the accuracy score in a list
#  called Accuracy_Scores. The code then prints the accuracy score of each model on the test data and 
# identifies the best-performing model based on the highest accuracy score. Finally, it prints a table 
# containing the accuracy score of each model and the best model among them.


highest_accuracy = PrettyTable(['Model', 'Accuracy'])
highest_accuracy.add_row([model_names[index], Accuracy_Scores[index]])
print(highest_accuracy)
print("\nTrain Dataset Features in form of Dataframe:","\t\t", " Test Dataset Features in form of Dataframe:")
print("============","\t\t"," ============\n")
print_side_by_side(label_encoded_train_dataset, label_encoded_test_dataset)
# This code block is displaying the detailed performance of all the models by creating a pretty table 
# that contains the accuracy scores of each model. It also identifies the best model based on the highest 
# accuracy score obtained. It then creates another pretty table with the name of the best model and its 
# corresponding accuracy score. Finally, it displays the train and test dataset features in a side-by-side format.


Complete_dataset_train_test = pd.concat([label_encoded_train_dataset, label_encoded_test_dataset])
print("\n\n\nAll Train and Test Dataset Features in form of DataFrame:")
print("====================\n")
print(Complete_dataset_train_test)
X_all_features = Complete_dataset_train_test.loc[:, training_features]
y_all_labels =Complete_dataset_train_test.loc[:, target]
random_forest_classifier.fit(X_all_features,np.ravel(y_all_labels))
pickle.dump(random_forest_classifier, open('RandomForestClassifier.pkl', 'wb'))
classifier = pickle.load(open('RandomForestClassifier.pkl', 'rb'))
# This block of code concatenates the label-encoded training dataset and label-encoded 
# testing dataset using the pd.concat() function to create a new dataset with all the
#  features. Then, the input features and output labels are extracted from the concatenated
#  dataset into X_all_features and y_all_labels, respectively. Next, a RandomForestClassifier() 
# model is fit to the concatenated dataset using all the available features (X_all_features) 
# and their corresponding labels (y_all_labels). The fitted model is saved as a pickle file 
# using the pickle.dump() function. Finally, the model is loaded back into memory using the
#  pickle.load() function and stored in the variable classifier. This is useful when you want 
# to use the same model to make predictions on new data, without having to train the model again.


height = float(input("Please enter your Height here (centimeter): ").strip())
weight = int(input("Please enter your Weight here (kg):").strip())
hair = input("Please enter your Hair Length here (Bald/Long/Short/Medium): ").strip()
beard = input("Do you have beard? (Yes/No): ").strip()
scarf = input("Do you wear Scarf? (Yes/No): ").strip()
# this code prompts the user to input their height, weight, hair length, beard presence, and scarf presence through the console. The inputs are then stored in the corresponding variables height, weight, hair, beard, and scarf. input() is a built-in Python function that allows the program to accept user input from the console. The strip() method is used to remove any leading or trailing white spaces from the input string. float() and int() are used to convert the input values for height and weight respectively to the appropriate data type.


user_input_df = pd.DataFrame({'height':[height], 'weight':[weight],'hair': [hair],'beard': [beard],'scarf': [scarf]})
user_input_df = user_input_df[['height', 'weight','hair','beard','scarf']]
print("\nUser input in Actual DataFrame form:")
print("=================\n")
print(user_input_df)
user_input_df['scarf']=labelEncoder_scarf.fit_transform(user_input_df['scarf'])
user_input_df['beard']=labelEncoder_beard.fit_transform(user_input_df['beard'])
user_input_df['hair']=labelEncoder_hair.fit_transform(user_input_df['hair'])
# This code creates a Pandas DataFrame user_input_df to store the input data from 
# the user. It includes the user's height, weight, hair length, beard presence, and 
# scarf presence. The DataFrame is then rearranged so that the columns are in a 
# specific order using user_input_df[['height', 'weight','hair','beard','scarf']]. 
# Finally, the categorical variables 'hair', 'beard', and 'scarf' are encoded using 
# LabelEncoder objects (labelEncoder_hair, labelEncoder_beard, labelEncoder_scarf) 
# that were previously defined in the code. The fit_transform() method is used to fit 
# the encoder to the input data and transform the categorical values to numerical values.


print("\n\n\nUser input in Encoded DataFrame form:")
print("========================\n")
print(user_input_df)
unseen_input_prediction = random_forest_classifier.predict(user_input_df)
t = PrettyTable([' ** Prediction ** '])
if(unseen_input_prediction ==1):
    t.add_row(['Male'])
if(unseen_input_prediction == 0):
    t.add_row(['Female'])
print(t)
# This code takes the user input for height, weight, hair length, beard, and scarf, and creates a
#  DataFrame from it. Then, it prints the DataFrame to show the user input in actual DataFrame form.
#  Next, it encodes the categorical variables (hair, beard, and scarf) using the LabelEncoder objects
#  that were fit earlier in the code. This creates an encoded version of the user input DataFrame.
#  Finally, the encoded user input is passed into the trained random forest classifier to make a
#  prediction of gender. The prediction is printed using a PrettyTable object. If the prediction is 1,
#  the table shows 'Male', and if the prediction is 0, it shows 'Female'.



