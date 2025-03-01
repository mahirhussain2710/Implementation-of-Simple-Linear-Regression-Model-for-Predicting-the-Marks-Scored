# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mahir Hussain S
RegisterNumber:212223040109
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
#Graph plot for training data
plt.scatter(x_train,y_train,color='orange')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='purple')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
df.head()

![Screenshot 2025-02-27 221752](https://github.com/user-attachments/assets/96bba0e6-1ece-4472-a24f-d5ad51b92bb2)

df.tail()

![Screenshot 2025-02-27 221931](https://github.com/user-attachments/assets/a5518e0d-dede-4fd2-84c4-6a6a79c356f3)

Array value of X

![Screenshot 2025-02-27 222030](https://github.com/user-attachments/assets/057289f0-6d70-401b-a6c2-e7cc968db79e)

Array value of Y

![Screenshot 2025-02-27 222130](https://github.com/user-attachments/assets/458f8cc9-9278-4722-8ed7-2ccf558b5dfd)

Values of Y prediction

![Screenshot 2025-02-27 222231](https://github.com/user-attachments/assets/bbd92025-1b36-4008-8f28-b1cec503721b)

Array values of Y test

![Screenshot 2025-02-27 222317](https://github.com/user-attachments/assets/cdc32b64-3192-4a66-a90d-ec876205af40)

Values of MSE, MAE and RMSE

![Screenshot 2025-02-27 222558](https://github.com/user-attachments/assets/2a383c7a-2e89-46d3-86cd-2a8f0ec090b5)

Training Set Graph

![Screenshot 2025-02-27 222441](https://github.com/user-attachments/assets/7286e0b3-630b-4080-bf5b-24f6292bf2c2)

Test Set Graph

![Screenshot 2025-02-27 162901](https://github.com/user-attachments/assets/9fc83ea4-cf74-4a62-a677-cde8632fb218)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
