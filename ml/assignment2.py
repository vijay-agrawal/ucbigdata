#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ml'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
#### ASSIGNMENT 2: PREDICT BOSTON HOUSING VALUES USING Multi Linear Regression

#%%
import pandas as pd

#%% [markdown]
# ### STEPS
# * ENSURE HEALTH OF DATA - Remove any rows with missing values, outliers
# * ENSURE DISTRIBUTION OF DATA - ensure data distribution is normal. If not apply transformations/mutations
# * CHOOSE MODEL - Use OLS if independent variables have a linear relation with dependent variable
# * VERIFY ASSUMPTIONS OF THE MODEL
# * ITERATE - Based on results, fine tune the model
# * VALIDATE - Validate with test data
# * ITERATE SOME MORE - If Test Data r-sqrd not as per expectation, model might be over-fit. Tune it further

#%%
boston_housing_df = pd.read_csv("boston.csv")
boston_housing_df.head()


#%% [markdown]
# Validate the data types are proper (float/int for numeric types)
#%%
boston_housing_df.dtypes

#%% [markdown]
# Data types are verified to be correct - floats and ints for all numbers

#%%
boston_housing_df.shape

#%% [markdown]
# Note the number of rows - 
#%% [markdown]
# Analyze Basic stats to see std-dev, mean etc
#%%
pd.DataFrame.describe(boston_housing_df)

#%% [markdown]
# It is observed from the stats that there are outliers in CRIM column. mean is .26 but max is 88!
# We will use zscore to remove outliers
#%%
from scipy import stats

#%% [markdown]
# Remove outliers
#%%
import numpy as np 
boston_housing_df = boston_housing_df[(np.abs(stats.zscore(boston_housing_df))<3).all(axis=1)]

#%%
pd.DataFrame.describe(boston_housing_df)

#%% [markdown]
# As can be seen, the max is now much closer, for CRIM column, to the mean
# Around 40 rows out of 506 were removed (~10%) - thats acceptable

#%%
boston_housing_df.shape

#%% [markdown]
# Check if there are any nulls in data
#%%
sns.heatmap(boston_housing_df.isnull(), cbar=False)

#%% [markdown]
# No null values are observed as there is no white bars in the heatmap

#%% [markdown]
#### CHECK FOR MULTICOLLINEARITY, DISTRIBUTION

#%%
import seaborn as sns 

#%% [markdown]
# Observe the distribution of Median Value variable

#%%
sns.set(color_codes=True)

#%%
sns.set(style='white', palette='muted')
#%%
sns.distplot(boston_housing_df['MV'])

#%% [markdown]
# As can be seen, the plot is more or less a good bell curve, not much skewness
# This meets one of the conditions for Linear Regression Modeling
# Let us now examine collinearity

#%%
sns.pairplot(boston_housing_df)

#%% [markdown]
# As can be seen, <br/>
# 1) Relationship of most variables except RM (Number of Rooms) is not as linear as desired <br/>
# 2) Variables CRIM, DIS, AGE, NOX are heavily skewed as can be seen in the historgram of these variables <br/>
# 

#%% [markdown]
# Let us also examine the correlation coefficients
#%%
corr = np.round(boston_housing_df.corr(), decimals=2)

#%%
corr.style.background_gradient(cmap='seismic')

#%%
corr.to_csv('corr4.csv')

#%% [markdown]
# As can be seen in the correlation csv, many of the coeeficients are in the 0.2 to 0.4 (in the Median Value row) <br/>
#  indicating there is no strong correlation between the target variable and independent variables <br/>
# NOX and INDUS are highly correlated (correlation coefficient is 0.75 - anything >0.6 is considered high) <br/>
# NOX and DIS are highly correlated (-0.83)

#%% [markdown]
# To mitigate the skewness, let us apply log transformations to the variables


#%%
boston_housing_df['log_CRIM'] = np.log10(boston_housing_df['CRIM'])
boston_housing_df = boston_housing_df.drop(columns = ['CRIM'])
boston_housing_df['log_DIS'] = np.log10(boston_housing_df['DIS'])
boston_housing_df = boston_housing_df.drop(columns = ['DIS'])
boston_housing_df['log_AGE'] = np.log10(boston_housing_df['AGE'])
boston_housing_df = boston_housing_df.drop(columns = ['AGE'])
boston_housing_df['log_NOX'] = np.log10(boston_housing_df['NOX'])
boston_housing_df = boston_housing_df.drop(columns = ['NOX'])


#%% [markdown]
### APPLY THE MODEL
# We are now ready to apply the model. <br/>
# First, create train and test data sets. Approx 20% data is left as test data.

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(boston_housing_df.drop("MV", axis=1), boston_housing_df['MV'], test_size = 0.2, random_state = 112)

#%%
import statsmodels.api as sm 


#%%
X_train = sm.add_constant(X_train)

#%% [markdown]
# Apply the model

#%%
my_model = sm.OLS(Y_train, X_train)

#%%
result=my_model.fit()

#%%
print(result.summary())

#%% [markdown]
# ANALYSIS <br/>
# R-sqrd: goodness of fit - is 0.67 - this is acceptable <br/>
# Adjusted R-sqrd is also close (so we dont have too many variables) <br/>
# Adjusted R-sqrd: penalizes for too many predictors <br/>
# F-statistic:  Prob(F-statistic) is < 0.05, predictors are doing good <br/>
# AIC = log likelihood+constant - minimize AIC <br/>
# BIC = same as AIC but it penalizes for too many predictors. BIC will always be slightly higher than AIC <br/>
#

#%% [markdown]
# Observe the P values in the results for each variable. <br/>
# For INDUS, B, the P value is > 0.05 - so this variable is not significant - remove it
# Also observe the correlation co-efficients. Columns with very low coeffs could be removed

#%%
X_train.drop(['INDUS', 'B'], axis=1, inplace = True)


#%% [markdown]
# Re-run the model

#%%
my_model2 = sm.OLS(Y_train, X_train)

#%%
result2 = my_model2.fit()

#%%
print (result2.summary())

#%% [markdown]
# No significant improvement observed in the results. 

#%% [markdown] 
### VALIDATE ON TEST DATA
#%%
from sklearn.metrics import r2_score


#%%
X_test = X_test.drop(['INDUS', 'B'], axis=1)

#%%
predictions = result2.predict(sm.add_constant(X_test))

#%%
r2_score(Y_test, predictions)

#%% [markdown]
# R-sqrd value on test dataset is better than train dataset. <br/>
# Model is not overfit and we can recommend using this model <br/>
# Additional tuning can be done by removing skewness, removing specific columns based on correlation co-efficients <br/>
# 