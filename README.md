# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.1.Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeClassifier from sklearn and apply the model on the dataset. 
5.Predict the values of array. 
6.Import metrics from sklearn and calculate the accuracy of the model on the dataset. 
7.Predict the values of array.
8.Apply to new unknown values.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ashish g
RegisterNumber:  212221240007

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics   
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

## Output:

DATA HEAD
![1](https://user-images.githubusercontent.com/95471278/174229612-a1960e57-2ee1-40cf-9de6-4ddec29b30f8.png)

DATA INFO

![2](https://user-images.githubusercontent.com/95471278/174229912-14180ce7-b08b-4a25-91f3-37a39fd8a93b.png)

DATA ISNULL

![3](https://user-images.githubusercontent.com/95471278/174230296-2df74d00-989c-4bbd-8f9d-72231417055e.png)

DATA LEFT
![0](https://user-images.githubusercontent.com/95471278/174230360-025a295e-1c53-4561-9306-7d653e0ffe5f.png)

X HEAD
![5](https://user-images.githubusercontent.com/95471278/174230498-29fc3740-3d15-4672-b52a-58685573a0ce.png)

DATA FIT
![6](https://user-images.githubusercontent.com/95471278/174230629-d2e15cbd-fff7-40f8-bd27-e7ee7095b25a.png)

ACCURACY

![7](https://user-images.githubusercontent.com/95471278/174230864-8440836c-5545-49f7-96bb-8a5ac33176a4.png)

PREDICTED VALUES
![8](https://user-images.githubusercontent.com/95471278/174231009-94043b6b-f5fd-4dec-8a78-0dd7bdeebb6a.png)

Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

