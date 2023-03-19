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


print("\n\n\nAttributes Names in Train Dataset:")
print("==================================\n")
print(train_dataset.columns)
print("\n\n\nNumber of instances in Train Dataset:")
print("====================================")
print("Train Data instances: ",train_dataset.shape[0])


print("\nTrain Dataset:")
print("==============\n")
train_dataset.columns.name = 'index'
print(train_dataset)


print("\n\n\nAttributes Names in Test Dataset:")
print("==================================\n")
print(test_dataset.columns)
print("\n\n\nNumber of instances in Test Dataset:")
print("====================================")
print("Test Data instances: ",test_dataset.shape[0])


print("\nTest Dataset:")
print("==============\n")
test_dataset.columns.name = 'index'
print(test_dataset)


train_Data_Label_Male = train_dataset[train_dataset['gender']== 'Male']
print("\n Train instances having label 'Male': \"",train_Data_Label_Male.shape[0],"\"")
train_Data_Label_Male.columns.name = 'index'
print("================\n")
print(train_Data_Label_Male)


train_Data_Label_Female = train_dataset[train_dataset['gender'] == 'Female']
print("\n Train instances having label 'Female': \"",train_Data_Label_Female.shape[0],"\"")
train_Data_Label_Female.columns.name = 'index'
print("=========================\n")
print(train_Data_Label_Female)


test_Data_Label_Male = test_dataset[test_dataset['gender'] == 'Male']
print("\n Test instances having label 'Male': \"",test_Data_Label_Male.shape[0],"\"")
test_Data_Label_Male.columns.name = 'index'
print("=================================\n")
print(test_Data_Label_Male)


test_Data_Label_Female = test_dataset[test_dataset['gender'] == 'Female']
print("\n Test instances having label 'Female': \"",test_Data_Label_Female.shape[0],"\"")
test_Data_Label_Female.columns.name = 'index'
print("==================================\n")
print(test_Data_Label_Female)


print("\n Visual representation of total number of 'Males' and 'Females' in Train Dataset:")
print("===========================\n")
train_data_gender_grouped = train_dataset.groupby('gender')['gender'].count().plot(kind = 'bar', rot = 0)
train_data_gender_grouped.set(xlabel = 'Gender')
train_data_gender_grouped.legend(['Frequency'], loc='upper right')


print("\n Visual representation of total number of 'Males' and 'Females' in Test Dataset:")
print("===========================\n")
test_data_gender_grouped = test_dataset.groupby('gender')['gender'].count().plot(kind = 'bar', rot = 0)
test_data_gender_grouped.set (xlabel = 'Gender')
test_data_gender_grouped.legend (['Frequency'], loc='upper right')


print("\n Number of people having various hair length in Train dataset:")
print("======================\n")
train_dataset.groupby('hair')['hair'].count().sort_values(ascending=False).plot(kind = 'bar', color = 'brown')


print("\n Number of people having various hair length in Test dataset:")
print("======================\n")
test_dataset.groupby('hair')['hair'].count().sort_values(ascending=False).plot(kind = 'bar', color = 'plasma')


print("\n Number of people have/haven't beard in Train dataset:")
print("==========================\n")
train_dataset.groupby('beard')['beard'].count().sort_values(ascending=False).plot(kind ='bar', colormap = 'RdGy_r')


print("\n Number of people have/haven't beard in Test dataset:")
print("==========================\n")
test_dataset.groupby('beard')['beard'].count().sort_values(ascending=False).plot(kind = 'bar', colormap = 'plasma_r')


train_dataset = train_dataset.fillna(' ')
Preprocessed_train_dataset = train_dataset.round({'height': 2})


def print_side_by_side(*objs, **kwds):
    ''' function print objects side by side '''
    from pandas.io.formats.printing import adjoin
    space = kwds.get('space', 15)
    reprs = [repr(obj).split('\n') for obj in objs]
    print(adjoin(space, *reprs))


print("\nTrain dataset before pre-processing:", "\t\t\t", "Train dataset after pre-processing:")
print("============","\t\t\t"," ============\n\n")
print_side_by_side(train_dataset ,Preprocessed_train_dataset)


test_dataset = test_dataset.fillna(' ')
Preprocessed_test_dataset = test_dataset.round({'height': 2})


print("\nTest dataset before pre-processing:", "\t\t\t", "Test dataset after pre-processing:")
print("============","\t\t\t"," ============\n\n")
print_side_by_side(test_dataset , Preprocessed_test_dataset)


labelEncoder_gender = LabelEncoder()
labelEncoder_scarf = LabelEncoder()
labelEncoder_hair = LabelEncoder()
labelEncoder_beard = LabelEncoder()


gender = ['Female','Male']
scarf = ['Yes','No']
beard = ['Yes','No']
hairLength = ['Bald','Long','Short','Medium']


labelEncoder_gender.fit(gender)
labelEncoder_scarf.fit(scarf)
labelEncoder_beard.fit(beard)
labelEncoder_hair.fit(hairLength)


label_encoded_train_dataset=Preprocessed_train_dataset.copy(deep=True)
label_encoded_test_dataset=Preprocessed_test_dataset.copy(deep=True)


print('\nGender attribute Label Encoding in Train dataset:')
print("=====================\n")
label_encoded_train_dataset['gender'] = labelEncoder_gender.fit_transform(Preprocessed_train_dataset['gender'])


Preprocessed_train_dataset['encoded_values_gender'] = label_encoded_train_dataset['gender']
print(Preprocessed_train_dataset[['gender','encoded_values_gender']])


print('\n\n\nScarf attribute Label Encoding in Train dataset:')
print("=====================\n")
label_encoded_train_dataset['scarf'] = labelEncoder_scarf.fit_transform(Preprocessed_train_dataset['scarf'])


Preprocessed_train_dataset['encoded_values_scarf'] = label_encoded_train_dataset['scarf']
print(Preprocessed_train_dataset[['scarf' , 'encoded_values_scarf']])


print('\n\n\nBeard attribute Label Encoding in Train dataset:')
print("=========================\n")
label_encoded_train_dataset['beard'] = labelEncoder_beard.fit_transform(Preprocessed_train_dataset['beard'])


Preprocessed_train_dataset['encoded_values_beard'] = label_encoded_train_dataset['beard']
print(Preprocessed_train_dataset[['beard','encoded_values_beard']])


print('\n\n\nHair attribute Label Encoding in Train dataset:')
print("======================\n")
label_encoded_train_dataset['hair'] = labelEncoder_hair.fit_transform(Preprocessed_train_dataset['hair'])


Preprocessed_train_dataset['encoded_values_hair']=label_encoded_train_dataset['hair']
print(Preprocessed_train_dataset[['hair','encoded_values_hair']])


label_encoded_test_dataset['gender']=labelEncoder_gender.fit_transform(Preprocessed_test_dataset['gender'])
label_encoded_test_dataset['scarf'] =labelEncoder_scarf.fit_transform (Preprocessed_test_dataset['scarf'])


label_encoded_test_dataset['beard']=labelEncoder_beard.fit_transform(Preprocessed_test_dataset['beard'])
label_encoded_test_dataset['hair']=labelEncoder_hair.fit_transform(Preprocessed_test_dataset['hair'])


training_features = ['height', 'weight', 'hair', 'beard', 'scarf']
target = 'gender'


X_train = label_encoded_train_dataset.loc[:, training_features]
y_train =label_encoded_train_dataset.loc[:, target]


X_test = label_encoded_test_dataset.loc[:, training_features]
y_test =label_encoded_test_dataset.loc[:, target]


model_names = []
logistic_regression = LogisticRegression()
print("\nParameters and their values:")


print("==================\n")
print(logistic_regression)


logistic_regression.fit(X_train,np.ravel(y_train))
model_names.append('LogisticRegression')


random_forest_classifier = RandomForestClassifier()
print("\nParameters and their values:")
print("======================\n")
print(random_forest_classifier)


random_forest_classifier.fit(X_train,np.ravel(y_train))
model_names.append('RandomForestClassifier')


linear_svc = LinearSVC()
print("\nParameters and their values:")
print("=====================\n")
print(linear_svc)


linear_svc.fit(X_train,np.ravel(y_train))
model_names.append('LinearSVC')


bernoulli_nb = BernoulliNB()
print("\nParameters and their values:")
print("======================\n")
print(bernoulli_nb)


bernoulli_nb.fit(X_train,np.ravel(y_train))
model_names.append('BernoulliNB')


Accuracy_Scores=[]
y_prediction = logistic_regression.predict(X_test)
Accuracy_Score = round(accuracy_score(y_test,
y_prediction),2)


inverse_transform_values = labelEncoder_gender.inverse_transform(y_prediction)
print("\nPrediction using Logistic Regression:")
print("=======================\n")
print(Preprocessed_test_dataset.assign(predicted_gender = inverse_transform_values))


print("\n\nAccuracy score : ")
print("===============")
print(Accuracy_Score)
Accuracy_Scores.append(Accuracy_Score)


y_prediction =random_forest_classifier.predict(X_test)
Accuracy_Score = round(accuracy_score(y_test,y_prediction),2)


inverse_transform_values = labelEncoder_gender.inverse_transform(y_prediction)
print("\nPrediction using RandomForestClassifier:")
print("========================\n")
print(Preprocessed_test_dataset.assign(predicted_gender = inverse_transform_values))


print("\n\nAccuracy score : ")
print("===============")
print(Accuracy_Score)
Accuracy_Scores.append(Accuracy_Score)


y_prediction = linear_svc.predict(X_test)
Accuracy_Score = round(accuracy_score(y_test,y_prediction),2)


inverse_transform_values = labelEncoder_gender.inverse_transform(y_prediction)
print("\nPrediction using LinearSVC:")
print("==========================\n")
print(Preprocessed_test_dataset.assign(predicted_gender = inverse_transform_values))


print("\n\nAccuracy score : ")
print("===============")
print(Accuracy_Score)
Accuracy_Scores.append(Accuracy_Score)


y_prediction = bernoulli_nb.predict(X_test)
Accuracy_Score = round(accuracy_score(y_test,y_prediction),2)


inverse_transform_values = labelEncoder_gender.inverse_transform(y_prediction)
print("\nPrediction using BernoulliNB:")
print("============================\n")
print(Preprocessed_test_dataset.assign(predicted_gender = inverse_transform_values))


print("\n\nAccuracy score : ")
print("===============")
print(Accuracy_Score)
Accuracy_Scores.append(Accuracy_Score)


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


highest_accuracy = PrettyTable(['Model', 'Accuracy'])
highest_accuracy.add_row([model_names[index], Accuracy_Scores[index]])
print(highest_accuracy)


print("\nTrain Dataset Features in form of Dataframe:","\t\t", " Test Dataset Features in form of Dataframe:")
print("============","\t\t"," ============\n")
print_side_by_side(label_encoded_train_dataset, label_encoded_test_dataset)


Complete_dataset_train_test = pd.concat([label_encoded_train_dataset, label_encoded_test_dataset])
print("\n\n\nAll Train and Test Dataset Features in form of DataFrame:")

print("====================\n")
print(Complete_dataset_train_test)

X_all_features = Complete_dataset_train_test.loc[:, training_features]
y_all_labels =Complete_dataset_train_test.loc[:, target]


random_forest_classifier.fit(X_all_features,np.ravel(y_all_labels))


pickle.dump(random_forest_classifier, open('RandomForestClassifier.pkl', 'wb'))


classifier = pickle.load(open('RandomForestClassifier.pkl', 'rb'))


height = float(input("Please enter your Height here (centimeter): ").strip())
weight = int(input("Please enter your Weight here (kg):").strip())
hair = input("Please enter your Hair Length here (Bald/Long/Short/Medium): ").strip()


beard = input("Do you have beard? (Yes/No): ").strip()
scarf = input("Do you wear Scarf? (Yes/No): ").strip()


user_input_df = pd.DataFrame({'height':[height], 'weight':[weight],'hair': [hair],'beard': [beard],'scarf': [scarf]})
user_input_df = user_input_df[['height', 'weight','hair','beard','scarf']]


print("\nUser input in Actual DataFrame form:")
print("=================\n")
print(user_input_df)


user_input_df['scarf']=labelEncoder_scarf.fit_transform(user_input_df['scarf'])
user_input_df['beard']=labelEncoder_beard.fit_transform(user_input_df['beard'])
user_input_df['hair']=labelEncoder_hair.fit_transform(user_input_df['hair'])


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



