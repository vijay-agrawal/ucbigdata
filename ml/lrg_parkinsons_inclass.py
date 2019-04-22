#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ml'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# TA CLASS 

#%%
import pandas as pd

#%% [markdown]
# ### PART 1: FIND OUT WHICH VARIABLES MAKE A BUSINESS SENSE TO KEEP
# HEALTH OF DATA - missing values, outliers
# DISTRIBUTION OF DATA - should be normal
# LOOK AT SCATTER PLOTS/CORR OF ALL VARIABLES - shows absence of linear corr..if scatter is curving around
# it gives hint on what kind of model/transformations to apply
# Above will give idea of what variables to keep, what transformations to apply
# 
# Lets take Motor Score as dependent variable. Total Score could also be..
# Others are independent variables/predictors/features

#%%
parkinsons_df = pd.read_csv("parkinsons_data.csv")
parkinsons_df.head(10)

#%% [markdown]
##### Drop the Variables that are clearly not needed: "subject#", 'total_UPDRS', 'test_time'


#%%
parkinsons_df.drop(['subject#', 'total_UPDRS', 'test_time'], axis=1, inplace=True)

#%% [markdown]
###### Check distribution of the dependent variables
# One of assumptions of linear model is that distribution of the feature variable is normal
# If not normal, use techniques such as transformation to normalize
# Also observe outliers, missing values (use seaborn sns heatmap)
# If there are many missing values, ignore the observations, or the feature itself
# If there are only some missing values, check if you can impute them
# If there are outliers, histogram will show them. Also z-score if data is normally distributed
# Also box plot shows outliers.Use box plots/quartile range etc to find thresholds beyond which you will
# call it outlier..then filter by those values and remove them..(10th, 90th quartile, for example)
# Regression model is not very sensitive to the threshold number - anything between 85-95 should work..
#%%
import numpy as np
np.histogram(parkinsons_df['motor_UPDRS'], bins=30)

#%%
corr = np.round(parkinsons_df.corr(), decimals=2)

#%%
corr.style.background_gradient(cmap='seismic')

#%%
corr.to_csv('corr2.csv')

#%% [markdown]
# Watch for corr coefficients..Value > 0.6 generally means high correlation <br/>
# High value means there is linear correlation, but low value does not mean there is no correlation
# As can be seen in output, for motor_UPDRS, only age has a slight correlation, all other fields
# are weak. 
# Also look at corr coefficients between the independent variables
# Shimmer variable is highly correlated with Shimmer(db), Shimmer APQ3 etc..remove ones that are not 
# relevant from business standpoint
# 
#%% [markdown]
# Also look at scatter plot of all correlations
# That will also give you an idea of which ones to keep
# ANALYSIS
# R-sqrd: goodness of fit --- mean sqrd error/sum of sqrd errors -1: more the variables, better the r-sqrd
# Adjusted R-sqrd: penalizes for too many predictors
# F-statistic: How good is your model compared to no-predictor/null hypothesis..Based on chi-sqrd. 
# If Prob(F-statistic) is < 0.05, predictors are doing good
# Log likehood: How likely are you to generate your data back by using the computed set of co-efficients
# If it increases, it is better..good to compare it 
# AIC = log likelihood+constant - minimize AIC
# BIC = same as AIC but it penalizes for too many predictors. BIC will always be slightly higher than AIC
#
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(parkinsons_df.drop("motor_UPDRS", axis=1), parkinsons_df['motor_UPDRS'], test_size = 0.2, random_state = 112)

#%%
import statsmodels.api as sm 


#%%
X_train = sm.add_constant(X_train)

#%% [markdown]
# Ordinary Least Squres (OLS) 

#%%
my_model = sm.OLS(Y_train, X_train)

#%%
result=my_model.fit()

#%%
print(result.summary())

#%% [markdown]
# Observe the P values in the results for each variable.
# In previous steps, we found HNR, Jitter(%) to be highly correlated with others. Remove them
# Also any variable with p-value > 0.05, remove it
X_train.drop(['Jitter(%)', 'Shimmer', 'HNR'], axis=1, inplace = True)


#%%
my_model2 = sm.OLS(Y_train, X_train)

#%%
result2 = my_model2.fit()

#%%
print (result2.summary())

#%% [markdown]
# R-sqrd became less as expected. But adjusted R-sqrd also decreased
# Normally, when right variables are removed, AIC, BIC should reduce..
# When removing insignificant variables, drop in R-sqrd should not be significant

# R-sqrd is low for this model (14%) which means linear model is not a good solution for this 
# type of data.
# Since variables are highly correlated, maybe decision tree is a better model
#

#%% [markdown]
# Heteroskedasticity: refers to the circumstance in which the variability of a variable is unequal across the range of values of a second variable that predicts it
# Tries to measure spread of data being constant for all variables. 
# so basically the histogram is similar for all variables
# residual: difference between actual Y value and predicted Y value
# residual plot should have values centered around zero
# thickness of the spread should be constant - this is what Heterskedasticity is measuring

#%% [markdown]
### Now use the model on the test dataset
# R-sqrd should be similar
# If r-sqrd lower, model is overfitted. Remove variables that are insignificant..



#%%
import seaborn as sns 

#%%
sns.pairplot(parkinsons_df)

#%%
sns.heatmap(parkinsons_df.isnull(), cbar=False)

#%%
