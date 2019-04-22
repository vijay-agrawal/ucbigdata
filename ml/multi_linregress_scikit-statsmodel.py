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
# ## MULTIPLE LINEAR REGRESSION USING SCIKIT-LEARN

#%%
marketing_df = pd.read_csv("marketing_data.csv")
marketing_df.head()
marketing_df.shape

#%% [markdown]
# split data into train and test

#%%
from sklearn.model_selection import train_test_split

#%%
X_train, X_test, Y_train, Y_test = train_test_split(marketing_df.drop("sales", axis=1), marketing_df['sales'], test_size=0.2, random_state=112)

#%% [markdown]
# import LinearRegression class

#%%
from sklearn.linear_model import LinearRegression

#%% [markdown]
# create an object of LinearRegression class

#%%
marketing_model = LinearRegression(normalize=True)

#%% [markdown]
# make predictions on the test set
marketing_model.fit(X_train, Y_train)



#%%
predictions = marketing_model.predict(X_test)

#%%
pd.DataFrame({'Actual Value': Y_test, 'Prediction': predictions}).sample(5)

#%%
marketing_model.score(X_test, Y_test)

#%%
marketing_model.coef_

#%%
marketing_model.intercept_


#%% [markdown]
### MULTIPLE LR USING STATSMODELS


#%%
import statsmodels.api as sm 


#%%
X_train = sm.add_constant(X_train)

#%%
marketing_model = sm.OLS(Y_train, X_train)

#%%
result = marketing_model.fit()

#%%
print(result.summary())

#%% [markdown]
### Observations
# From the model results we see that the the parameter for newspaper is not significantly different from zero at 5% significance level.<br/>
# (because the confidence interval for its coefficient includes zero and the p-value is very high than the significance level of 0.05)<br/>
# Parameter for youtube and facebook variables, on the other hand, are significantly different from zero. (because the corresponding confidence interval includes zero and the p-value is almost zero.(lower than 0.05 significane level)<br/>
# Hence we can conclude that the money spent on newspaper advertising doesn't play a significant role in sales.<br/>
# Let's now test our inference and drop the newspaper variable and just use youtube and facebook advertisement data to predict the sales.


#%%
X_train = X_train.drop("newspaper", axis=1)

#%%
upd_model_result = sm.OLS(Y_train, X_train).fit()

#%%
print(upd_model_result.summary())

#%% [markdown]
# We see that even though our  R2  value has very slightly decreased, but the Adjusted  R2  value has increased. <br/>
# This confirms that the weight variable wasn't significant in calculating the blood fat content and by keeping the weight variable, the adjusted  R2  value was penalized. <br/>
# Let's also see the accuracy ( R2 ) value on the test set, with this updated model.

#%%
from sklearn.metrics import r2_score


#%%
X_test = X_test.drop("newspaper", axis=1)

#%%
predictions = upd_model_result.predict(sm.add_constant(X_test))

#%%
r2_score(Y_test, predictions)

#%%
