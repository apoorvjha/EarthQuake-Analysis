#!/usr/bin/env python
# coding: utf-8




import pandas as pd                # importing & preprocessing of data
import numpy as np                 # to perform mathematical operations
import matplotlib.pyplot as plt    # data visualization
from mpl_toolkits import mplot3d
import warnings





warnings.filterwarnings('ignore')





# importing data from CSV file into pandas dataframe
dataset=pd.read_csv("database.csv")





# dataset structure
print("Column Name",end="")
for i in range(35-len("Column Name")):
    print(end=" ")
print(" Data Type")
print("-------------------------------------------------------------")
for col in dataset.columns:
    print(col,end="")
    for i in range(30-len(col)):
        print(end=" ")
    print("|   ",type(dataset[str(col)][0]))





# dataset description
dataset.describe()





# correlation analysis of the dataset
dataset.corr()





dataset.head()





dataset.tail()


# # Observations from data exploration
# 
#    1. Magnitude is our target variable.
#    2. The average error in recording of magnitude is 0.07. Which means we hope to predict the value of dependent target variable with accuracy of one decimal place. 
#    3. The correlation of target variable with respect to other columns is our criterion to select the independent variable.
#    4. As we have 23,412 rows worth of data which is not enough to train any deep learning model with much of significant accuracy boost. So we are going to use a simple Random Forest regression model for predicting the value of our target variable.
#    5. As part of intution we have encorporated Latitude and Longitude columns in our independent variables even though their correlation is less as compared to others as climate is very intrinsically dependent upon the topological instance of the area.




# eliminating NANs present in the dataset

values=dataset.mean(skipna=True).to_dict()    # fetches the column-wise means and converts it into a dictionary  
dataset=dataset.fillna(value=values)





# preprocessing date and time

date=dataset['Date'].values
day=[]
month=[]
year=[]
for i in range(len(date)):
    if len(date[i])>10:
        temp1=date[i].split('T')
        temp=temp1[0].split('-')
        y=temp[0]
        m=temp[1]
        d=temp[2]
    else:
        temp=date[i].split('/')
        y=temp[2]
        m=temp[0]
        d=temp[1]
    month.append(int(m))
    day.append(int(d))
    year.append(int(y))
    
time=dataset['Time']
hour=[]
minute=[]
seconds=[]
for i in range(len(time)):
    if len(date[i])>10:
        temp1=time[i].split('T')
        temp=temp1[1].split(':')
        h=temp[0]
        m=temp[1]
        s=temp[2].split('Z')[0]
    else:
        temp=time[i].split(':')
        h=temp[0]
        m=temp[1]
        s=temp[2]
    hour.append(int(h))
    minute.append(int(m))
    seconds.append(float(s))





# splitting the columns of the dataset into dependent and independent sets.

independentSet=['Date','Time','Latitude','Longitude','Depth','Horizontal Distance']
X=[]               # independent variable
Y=[]               # dependent variable
latitude=dataset['Latitude'].values
longitude=dataset['Longitude'].values
depth=dataset['Depth'].values
hd=dataset['Horizontal Distance'].values
for i in range(len(latitude)):
    temp=[]
    for feature in independentSet:
        if feature=='Date':
            temp.append(day[i])
            temp.append(month[i])
            temp.append(year[i])
        elif feature=='Time':
            temp.append(hour[i])
            temp.append(minute[i])
            temp.append(seconds[i])
        elif feature=='Latitude':
            temp.append(latitude[i])
        elif feature=='Longitude':
            temp.append(longitude[i])
        elif feature=='Depth':
            temp.append(depth[i])
        else:
            temp.append(hd[i])
    X.append(temp)
Y=dataset['Magnitude'].values
X=np.array(X)
Y=np.array(Y)





# Visualization
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.view_init(15,45)
ax.set_title('Cluster Relationship between Magnitude, Latitude and Longitude')
ax.set_ylabel('Longitude')
ax.set_xlabel('Latitude')
ax.set_zlabel('Magnitude')
ax.scatter3D(latitude[:100],longitude[:100],Y[:100],c=longitude[:100],cmap='Greens')


# ### There is a special region where the clustered combination of latitude and longitude results into significant magnitude.




fig=plt.figure()
ax=plt.axes(projection='3d')
ax.view_init(15,45)
ax.set_title('Variation Relationship between Magnitude, Latitude and Longitude')
ax.set_ylabel('Longitude')
ax.set_xlabel('Latitude')
ax.set_zlabel('Magnitude')
ax.plot3D(latitude[:100],longitude[:100],Y[:100],'blue')


# ### As pointed out earlier that correlation of Magnitude is less with respect to Latitude ans longitude which can easily deduced from the plot above. Overall large variations are there which does not seem to depend in a particular fashion but on small steps we can see some straight lines which motivated us to incorporate as a feature in our analysis. 




# visualization
plt.xlabel('Depth')
plt.ylabel('Magnitude')
plt.title('Magnitude vs Depth')
plt.scatter(depth[:100],Y[:100])


# ### Magnitude seems to be dependent upon Depth only over a specific region which means we need to deploy a non linear model to completely fit this data as we have observed similar conclusion on 3D scatter plot of Magnitude vs Latitude and longitude.




# visualization
plt.xlabel('Horizontal Depth')
plt.ylabel('Magnitude')
plt.title('Magnitude vs Horizontal Depth')
plt.scatter(hd[:100],Y[:100])


# ### According to this plot it is clear that magnitude varies very significantly over a very small variation in horizontal depth. 




ttsRatio=0.3      # trainning sample ratio with respect to total data
X_train=X[:int(ttsRatio * len(X))]
Y_train=Y[:int(ttsRatio * len(Y))]
X_test=X[int(ttsRatio * len(X)):]
Y_test=Y[int(ttsRatio * len(Y)):]





# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()





model.fit(X_train,Y_train)





predicted=model.predict(X_test)





predictionPerformance=model.score(X_test,Y_test)
print('R^2 score : ',predictionPerformance)





# multiple polynomial regression model
from sklearn.preprocessing import PolynomialFeatures 
poly = PolynomialFeatures(degree = 4) 
X_poly_train=poly.fit_transform(X_train) 
poly.fit(X_poly_train,Y_train)
X_poly_test=poly.fit_transform(X_test)
poly.fit(X_poly_test,Y_test)





from sklearn.linear_model import LinearRegression
model=LinearRegression()





model.fit(X_poly_train,Y_train)





prediction=model.predict(X_poly_test)





predictionPerformance=model.score(X_poly_test,Y_test)
print('R^2 score : ',predictionPerformance)


# ### Major disadvantage of polynomial regression is it's high sensitivity towards outliers. As outliers in the data tend to increase then then accuracy of the model suffers by the large margin as we can see from the above calculated R^2 score which is highly negative.  




# support vector machine regressor
from sklearn.svm import SVR
model=SVR()





model.fit(X_train,Y_train)





predicted=model.predict(X_test)





predictionPerformance=model.score(X_test,Y_test)
print('R^2 score : ',predictionPerformance)


# ## Conclusion
#   1. Polynomial regression has large degradation in it's performance due to presence of outliers in data.
#   2. Support vector machine regression is best performing model which supports the notion popular in data science community that SVMs perform best for applications where there is lack of much data to train deep learning models.
#   3. With proper tuning of parameters a much more concrete model can be formed which is our agenda for future.
