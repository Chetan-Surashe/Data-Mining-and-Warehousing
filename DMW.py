import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

"""#*Data collection and processing*"""

#Loading csv data to a pandas dataframe
heart_data=pd.read_csv(r"C:\Users\Acer\Downloads\Hdata.csv")

#Printing first 5 rows from dataset
heart_data.head()

#print last 5 rows of dataset
heart_data.tail()

#number of rows(number of records/samples) and columns(features) in the dataset
heart_data.shape

#getting some info about the data
heart_data.info()

#checking missing value
heart_data.isnull().sum()

#statistical measures about the data
heart_data.describe()

"""1. In the above data **count** represent number of datapoints in each column.  

2. **mean** --> represent mean value of the column

3. **std deviation**

4. **min** --> minimum value from the column


5. **25%,50%,75% percentile values**

**25th Percentile** - Also known as the first, or lower, quartile. The 25th percentile is the value at which 25% of the answers lie below that value, and 75% of the answers lie above that value.

**50th Percentile** - Also known as the Median. The median cuts the data set in half.  Half of the answers lie below the median and half lie above the median.

**75th Percentile** - Also known as the third, or upper, quartile. The 75th percentile is the value at which 25% of the answers lie above that value and 75% of the answers lie below that value. Above the 75th or below the 25th percentile - If your data falls above the 75th percentile or below the 25th percentile we still display your data and include a << or >> indicator noting that your club's position is above or below those points.

6. **max** --> maximum value from the column
"""

#Checking the distribution of Target Variable
heart_data['target'].value_counts()

"""**target--> 1 --> represents --> Heart Diesase detected**

**target--> 0 --> represents --> Heart Diesase not detected**

#splitting features and target
Here the target column is separated from all rest of the 13 features
"""

X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']

#printing X
print(X)

#printing Y
print(Y)

"""#Splitting the data into training data and testing data"""

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

"""test_size=0.2 -->indicates 20% data used for testing

stratify-->

random_state=2 -->
"""

print(X.shape,X_train.shape,X_test.shape)

"""X.shape-->the shape after removing the "target" column

X_train.shape-->shape of training data (80% data)

X_test.shape--> shape of testing data (20% data)

#Training the model
**By using logistic regression model**
"""

model=LogisticRegression()

#Training the LogisticRegression model with training data
model.fit(X_train,Y_train)#it will find relation between X_train & Y_train
#i.e. relation between "features" and "target"

"""#Model evaluation

**Accuracy score** --> it is nothing but The model will be ask to predict the "target" and this predicted value will be compared with the "original target" values

for eg. out of 100 records model is predicting 95 values so accuracy score-->95
"""

#Accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

print("Accuracy on training data: ",training_data_accuracy)

"""above value indicates the 85% out of 100 value is predicted correctly
above 75% is good accuracy
"""

#Accuracy on testing data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

print("Accuracy on test data: ",test_data_accuracy)

"""above value indicates the 80% out of 100 value is predicted correctly above 75% is good accuracy

#Building A Predictive System
**To determine the heart disease of new patient based on the trained data set**
"""

input_data=(34,0,1,118,210,0,1,192,0,0.7,2,0,2)

#change input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#we have to reshape the array--> since we have to predict it for 1 record at a time
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


prediction=model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
  print("The Person Does Not Have Heart Disease")
else:
  print("The Person Have Heart Disease")
  joblib.dump(model, 'DMW.pkl')
  