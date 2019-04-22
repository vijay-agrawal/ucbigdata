#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ml'))
	print(os.getcwd())
except:
	pass
#%%
import pandas as pd

#%% [markdown]
# ## SIMPLE LINEAR REGRESSION USING SCIKIT-LEARN

#%%
salary_df = pd.read_csv("simple linear regression data.csv")
salary_df.head(10)

#%% [markdown]
# split data into train and test

#%%
from sklearn.model_selection import train_test_split

#%%
X_train, X_test, Y_train, Y_test = train_test_split(salary_df.drop("Salary", axis=1), salary_df['Salary'], test_size=0.3, random_state=112)

#%% [markdown]
# import LinearRegression class

#%%
from sklearn.linear_model import LinearRegression

#%% [markdown]
# create an object of LinearRegression class

#%%
lrm = LinearRegression(normalize=True)

#%% [markdown]
# fit the lrm to the training set

#%%
lrm.fit(X_train, Y_train)

#%% [markdown]
# make predictions on the test set

predictions = lrm.predict(X_test)

#%%
pd.DataFrame({'Actual Value': Y_test, 'Prediction':predictions}).sample(5)

#%%
lrm.coef_

#%%
lrm.intercept_

#%%
lrm.score(X_test, Y_test)


#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#%%
mean_absolute_error(Y_test, predictions)

#%%
mean_squared_error(Y_test, predictions)

#%%
r2_score(Y_test, predictions)

#%% [markdown]
### LR WITH STATSMODELS

#%%
import statsmodels.api as sm 


#%%
X_train = sm.add_constant(X_train)

#%%
X_test = sm.add_constant(X_test)

#%%
sm_model = sm.OLS(Y_train, X_train)

#%%
result = sm_model.fit()

#%%
print(result.summary())

#%% [markdown]
# Get R-sqaured value

result.rsquared


#%%
result.rsquared_adj

#%%
result.pvalues

#%%
result.params

#%% [markdown]
# Get confidence Interval


#%%
result.conf_int()

#%% [markdown]
# Lets see accuracy value on test set
predictions = result.predict(sm.add_constant(X_test))


#%%
r2_score (Y_test, predictions)

#%% [markdown]
### LR WITH SciPy



#%%
from scipy.stats import linregress
import matplotlib.pyplot as plt
%matplotlib inline

#%%
regression_result = linregress(salary_df['WorkExperience'], salary_df['Salary'])

#%%
regression_result.slope

#%%
regression_result.intercept

#%%
plt.plot(salary_df['WorkExperience'], salary_df['Salary'], 'o', label='Original Data')
plt.plot(salary_df['WorkExperience'], \
    regression_result.intercept + regression_result.slope*salary_df['WorkExperience'], \
        'r', label='Fitted Line')
plt.xlabel("Work Experience (In Months)")
plt.ylabel("Salary (In thousand USD")
plt.legend()
plt.show()

#%%

#%%
