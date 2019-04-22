#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ml'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## UC ML ASSIGNMENT 1 - VIJAY AGRAWAL

#%%
import pandas as pd

#%% [markdown]
# ### PART 1: FEATURE ENGINEERING

#%%
nyc_sales_df = pd.read_csv("nyc-rolling-sales_clean.csv")
nyc_sales_df.head(10)

#%% [markdown]
# #### ADD YEAR OF SALE COLUMN

#%%
nyc_sales_df['SALE DATE'] = pd.to_datetime(nyc_sales_df['SALE DATE'])
nyc_sales_df['Year of Sale'] = nyc_sales_df['SALE DATE'].dt.year
nyc_sales_df.head(10)

#%% [markdown]
# ##### DROP SALE DATE COLUMN

#%%
nyc_sales_df.drop(['SALE DATE'], axis=1, inplace=True)
nyc_sales_df.head(10)

#%% [markdown]
# #### ONE HOT ENCODING FOR 'TAX CLASS' COLUMN

#%%
pd.get_dummies(nyc_sales_df, columns=['TAX CLASS AT PRESENT']).head(10)

#%% [markdown]
# ### PART 2: HYPOTHESIS TESTING

#%%
gestational_df = pd.read_csv("gestational_study.csv")
gestational_df


#%%
gestational_df['Gestational_Age'].corr(gestational_df['Birth_Weight'])

#%% [markdown]
# **A correlation co-efficient of 0.82 indicates strong correlatin between Gestational Age and Birth Weight**

#%%
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
splot=gestational_df.plot.scatter(x='Gestational_Age',
                      y='Birth_Weight',
                      c='DarkBlue')

#%% [markdown]
# **The scatter plot above reaffirms that the correlation is positive**

#%%
from scipy import stats


#%%
stats.pearsonr(gestational_df['Gestational_Age'], gestational_df['Birth_Weight'])

#%% [markdown]
# **Pearson correlation index is high indicating strong correlation The second value, P-value is very low (< 0.05) indicating the null hypothesis (that there is no significant correlation) stands rejected. P-value is extremely low, indicating > 99% confidence on a strong correlation**
