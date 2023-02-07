# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:23:38 2022

@author: 91995
"""

import pandas as pd
df = pd.read_csv("ToyotaCorolla (1).csv",encoding='latin1')
df.head()
df.dtypes
df.shape

#==================================================
# dropping the variables
df.drop("Id",axis=1,inplace=True)
df.drop("Model",axis=1,inplace=True)
df.drop("Mfg_Month",axis=1,inplace=True)
df.drop("Mfg_Year",axis=1,inplace=True)
df.drop("Fuel_Type",axis=1,inplace=True)
df.drop("Met_Color",axis=1,inplace=True)
df.drop("Color",axis=1,inplace=True)
df.drop("Automatic",axis=1,inplace=True)
df.drop("Cylinders",axis=1,inplace=True)
df.dtypes
df.shape

df.drop(df.iloc[:,9:],axis=1,inplace=True)
df.dtypes
df.head()
#=================================================
# Data visualization
# Histogram
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

for col in df:
    print(col)
    print(skew(df[col]))
    
    plt.figure()
    sns.distplot(df[col])
    plt.show()

#==================================================
# scatter plot
plt.scatter(df["Price"],df["Age_08_04"],color = "black")
plt.scatter(df["Price"],df["KM"],color = "black")
plt.scatter(df["Price"],df["HP"],color = "black")
plt.scatter(df["Price"],df["cc"],color = "black")
plt.scatter(df["Price"],df["Doors"],color = "black")
plt.scatter(df["Price"],df["Gears"],color = "black")
plt.scatter(df["Price"],df["Quarterly_Tax"],color = "black")
plt.scatter(df["Price"],df["Weight"],color = "black")
#============================================================
# boxplot
df.boxplot(column="Price",vert=False)
import numpy as np
Q1 = np.percentile(df["Price"],25)
Q2 = np.percentile(df["Price"],50)
Q3 = np.percentile(df["Price"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Price"]<LW) | (df["Price"]>UW)]
len(df[(df["Price"]<LW) | (df["Price"]>UW)])
# out layers are 34
df["Price"]=np.where(df["Price"]>UW,UW,np.where(df["Price"]<LW,LW,df["Price"]))
len(df[(df["Price"]<LW) | (df["Price"]>UW)])
# out layers are 0

df.boxplot(column="Age_08_04",vert=False)
import numpy as np
Q1 = np.percentile(df["Age_08_04"],25)
Q2 = np.percentile(df["Age_08_04"],50)
Q3 = np.percentile(df["Age_08_04"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Age_08_04"]<LW) | (df["Age_08_04"]>UW)]
len(df[(df["Age_08_04"]<LW) | (df["Age_08_04"]>UW)])
# out layers are 0


df.boxplot(column="KM",vert=False)
import numpy as np
Q1 = np.percentile(df["KM"],25)
Q2 = np.percentile(df["KM"],50)
Q3 = np.percentile(df["KM"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["KM"]<LW) | (df["KM"]>UW)]
len(df[(df["KM"]<LW) | (df["KM"]>UW)])
# out layers are 12
df["KM"]=np.where(df["KM"]>UW,UW,np.where(df["KM"]<LW,LW,df["KM"]))
len(df[(df["KM"]<LW) | (df["KM"]>UW)])
# out layers are 0


df.boxplot(column="HP",vert=False)
import numpy as np
Q1 = np.percentile(df["HP"],25)
Q2 = np.percentile(df["HP"],50)
Q3 = np.percentile(df["HP"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["HP"]<LW) | (df["HP"]>UW)]
len(df[(df["HP"]<LW) | (df["HP"]>UW)])
# out layers are 11
df["HP"]=np.where(df["HP"]>UW,UW,np.where(df["HP"]<LW,LW,df["HP"]))
len(df[(df["HP"]<LW) | (df["HP"]>UW)])
# out layers are 0


df.boxplot(column="Doors",vert=False)
import numpy as np
Q1 = np.percentile(df["Doors"],25)
Q2 = np.percentile(df["Doors"],50)
Q3 = np.percentile(df["Doors"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Doors"]<LW) | (df["Doors"]>UW)]
len(df[(df["Doors"]<LW) | (df["Doors"]>UW)])
# out layers are 0


df.boxplot(column="Gears",vert=False)
import numpy as np
Q1 = np.percentile(df["Gears"],25)
Q2 = np.percentile(df["Gears"],50)
Q3 = np.percentile(df["Gears"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Gears"]<LW) | (df["Gears"]>UW)]
len(df[(df["Gears"]<LW) | (df["Gears"]>UW)])
# out layers are 46
df["Gears"]=np.where(df["Gears"]>UW,UW,np.where(df["Gears"]<LW,LW,df["Gears"]))
len(df[(df["Gears"]<LW) | (df["Gears"]>UW)])
# out layers are 0

df.boxplot(column="Quarterly_Tax",vert=False)
import numpy as np
Q1 = np.percentile(df["Quarterly_Tax"],25)
Q2 = np.percentile(df["Quarterly_Tax"],50)
Q3 = np.percentile(df["Quarterly_Tax"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Quarterly_Tax"]<LW) | (df["Quarterly_Tax"]>UW)]
len(df[(df["Quarterly_Tax"]<LW) | (df["Quarterly_Tax"]>UW)])
# out layers are 224
df["Quarterly_Tax"]=np.where(df["Quarterly_Tax"]>UW,UW,np.where(df["Quarterly_Tax"]<LW,LW,df["Quarterly_Tax"]))
len(df[(df["Quarterly_Tax"]<LW) | (df["Quarterly_Tax"]>UW)])
# out layers are 0

df.boxplot(column="Weight",vert=False)
import numpy as np
Q1 = np.percentile(df["Weight"],25)
Q2 = np.percentile(df["Weight"],50)
Q3 = np.percentile(df["Weight"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Weight"]<LW) | (df["Weight"]>UW)]
len(df[(df["Weight"]<LW) | (df["Weight"]>UW)])
# out layers are 34
df["Weight"]=np.where(df["Weight"]>UW,UW,np.where(df["Weight"]<LW,LW,df["Weight"]))
len(df[(df["Weight"]<LW) | (df["Weight"]>UW)])
# out layers are 0
#======================================================================================
# Standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
df[["Price"]] = SS.fit_transform(df[["Price"]])
df[["Age_08_04"]] = SS.fit_transform(df[["Age_08_04"]])
df[["KM"]] = SS.fit_transform(df[["KM"]])
df[["HP"]] = SS.fit_transform(df[["HP"]])
df[["cc"]] = SS.fit_transform(df[["cc"]])
df[["Doors"]] = SS.fit_transform(df[["Doors"]])
df[["Gears"]] = SS.fit_transform(df[["Gears"]])
df[["Quarterly_Tax"]] = SS.fit_transform(df[["Quarterly_Tax"]])
df[["Weight"]] = SS.fit_transform(df[["Weight"]])

#=========================================================================================================
# Splitting the variables as X and Y

Y = df.iloc[:,:1]
X = df.iloc[:,1:]

# Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

# B0
LR.intercept_

# B1
LR.coef_

# predictions
Y_pred = LR.predict(X)
Y_pred

# Marics
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y, Y_pred)
print("Mean Squared Error :",mse.round(3))

import numpy as np
print("Root Mean Squared Error :", np.sqrt(mse).round(3))

r2 = r2_score(Y,Y_pred)
print("R square :", r2.round(3))
#===========================================================


import statsmodels.api as sma
Y_new =sma.add_constant(X)
lm2 = sma.OLS(Y,Y_new).fit()
lm2.summary()



#########################################################################


RSS =  np.sum((Y_pred-Y)**2)
Y_mean = np.mean(Y)
Y_mean
TSS = np.sum((Y-Y_mean)**2)
R2 = 1-(RSS/TSS)
print("R2:",R2)


vif = 1/(1-R2)
print("VIF value:",vif)

##########################################################################################

import matplotlib.pyplot as plt

import statsmodels.api as sm
qqplot=sm.qqplot(df,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()

list(np.where(df>10))

##########################################################################################


def get_standardized_values(vals):
    return(vals-vals.mean())/vals.std()

plt.scatter(get_standardized_values(df),
            get_standardized_values(df))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()









