# Code for the BUS803 Hypothesis Testing Lab Session


import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import t, f

# Load the data file
power_consumption = pd.read_excel('WeatherData.xlsx')

##########################################################################
# Model using Temperature only
##########################################################################

print("Create and analyse a model using temperature only")

# plot the data

plt.scatter(power_consumption["Temperature"],power_consumption["MWh"])
plt.xlabel("Temperature")
plt.ylabel("Mega-Watt Hours")
plt.title("Mega-Watt Hours v Temperature")
plt.grid()

plt.show()

# Fit a regression model

mdl_T = sm.OLS.from_formula('MWh ~ 1 + Temperature', power_consumption)
#在 statsmodels 的公式表达中，1 + 自变量 是显式地表明包含截距项。如果不写 1，截距项会被默认包含，但如果写 0 + 自变量，则会明确排除截距项。

results_T = mdl_T.fit()

print(results_T.summary())
##1. model summary
#R-squared: 0.548, which means 54.8% of the variation in the dependent variable (MWh) is explained by the independent variable (Temperature).
#Adjusted R-squared: 0.543, which adjusts the R-squared value for the number of predictors and sample size.
##2.coefficients
#Intercept: -26.5488, with a t-value of -2.971 and a p-value of 0.004. This indicates that the intercept is statistically significant.
#Temperature Coefficient: 3.8243, with a t-value of 10.900 and a p-value of 0.000. This indicates that temperature has a statistically significant positive relationship with MWh.
##3.model fit
#F-statistic: 118.8, with a p-value of 1.36e-18, indicates the overall model is statistically significant.
#Durbin-Watson Statistic: 1.898, suggesting there is minimal autocorrelation in the residuals (closer to 2 indicates no autocorrelation).
##4.normality of residuals
#The Omnibus test has a p-value of 0.212, and the Jarque-Bera test has a p-value of 0.280. These p-values suggest that the residuals do not deviate significantly from normality.
##5. Confidence Intervals:The 95% confidence interval for the Temperature coefficient is [3.128, 4.521], confirming the robustness of the effect.

# Overlay the fitted model onto the plot

# Get the extreme values of the independent data
x_lims = [power_consumption["Temperature"].min(),\
          power_consumption["Temperature"].max()]

# Create a dataframe
x_new = pd.DataFrame(x_lims, columns = ['Temperature'])

# Get predicted values
yhat = results_T.predict(x_new)

# Plot the original data
plt.scatter(power_consumption["Temperature"],\
            power_consumption["MWh"],label="raw data")
plt.xlabel("Temperature")
plt.ylabel("Mega-Watt Hours")
plt.title("Mega-Watt Hours v Temperature")
plt.grid()

# Overlay the model fit
plt.plot(x_lims,yhat,color="r",label="fitted model")

plt.legend()
plt.show()

######### If we want to calculate the fit statistics ourselves##################

# Calculate the sample variance
Y = power_consumption['MWh']
yhat = results_T.predict(power_consumption)
resids = (yhat-Y)
N = len(Y)
s2 = np.sum(resids**2)/(N-2)

print("The sample variance is {val:.6f}".format(val=s2))

# Calculate the standard errors
sBeta02 = s2*(np.sum(power_consumption["Temperature"]**2)\
              /(N*np.sum((power_consumption["Temperature"]-\
                          power_consumption["Temperature"].mean())**2)));
sBeta12 = s2/np.sum((power_consumption["Temperature"]-\
                     power_consumption["Temperature"].mean())**2)

print("The standard error for the constant is {val:.6f}"\
      .format(val=np.sqrt(sBeta02)))
print("The standard error for the Temperature is {val:.6f}"\
      .format(val=np.sqrt(sBeta12)))

# These can be obtained from the model fit using
results_T.bse**2

# Calculate the t-values
beta = results_T.params
t_beta0 = beta["Intercept"]/np.sqrt(sBeta02)
t_beta1 = beta["Temperature"]/np.sqrt(sBeta12)

print("The t-value for the constant is {val:.6f}".format(val=t_beta0))
print("The t-value for the Temperature is {val:.6f}".format(val=t_beta1))

# These can be obtained from the model fit using
results_T.tvalues

# Calulate the critical value
tc = t.ppf(0.025, N-2) 

print("The critical value for the t-value is {val:.6f}".format(val=tc))

# Calulate the p-values
pval0 = 2*(t.cdf(-np.abs(t_beta0), N-2))
pval1 = 2*(t.cdf(-np.abs(t_beta1), N-2))

print("The p-value for the constant is {val:.6f}".format(val=pval0))
print("The p-value for the Temperature is {val:.6f}".format(val=pval1))
#Intercept: -26.5488, with a t-value of -2.971 and a p-value of 0.004. This indicates that the intercept is statistically significant.
#Temperature Coefficient: 3.8243, with a t-value of 10.900 and a p-value of 0.000. This indicates that temperature has a statistically significant positive relationship with MWh.

# These can be obtained from the model fit using
results_T.pvalues

##########################################################################
# Model using Humidity only
##########################################################################

print("Create and analyse a model using humidity only")

# plot the data

plt.scatter(power_consumption["Humidex"],power_consumption["MWh"])
plt.xlabel("Humidex")
plt.ylabel("Mega-Watt Hours")
plt.title("Mega-Watt Hours v Humidex")
plt.grid()

plt.show()

# Fit a regression model

mdl_H = sm.OLS.from_formula('MWh ~ 1 + Humidex', power_consumption)
#在 statsmodels 的公式表达中，1 + 自变量 是显式地表明包含截距项。如果不写 1，截距项会被默认包含，但如果写 0 + 自变量，则会明确排除截距项。

results_H = mdl_H.fit()

print(results_H.summary())


# Overlay the fitted model onto the plot

# Get the extreme values of the independent data
x_lims = [power_consumption["Humidex"].min(),\
          power_consumption["Humidex"].max()]

# Create a dataframe
x_new = pd.DataFrame(x_lims, columns = ['Humidex'])

# Get predicted values
yhat = results_H.predict(x_new)

# Plot the original data
plt.scatter(power_consumption["Humidex"],\
            power_consumption["MWh"],label="raw data")
plt.xlabel("Humidex")
plt.ylabel("Mega-Watt Hours")
plt.title("Mega-Watt Hours v Humidex")
plt.grid()

# Overlay the model fit
plt.plot(x_lims,yhat,color="r",label="fitted model")

plt.legend()
plt.show()


#####################################################################
# Incorporate Humidity
#####################################################################

print("Create and analyse a model using temperature and humidex")

# plot the data

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(power_consumption["Temperature"], \
           power_consumption["Humidex"], \
           power_consumption["MWh"])

plt.xlabel("Temperature")
plt.ylabel("Humidex")
ax.set_zlabel("Mega-Watt Hours")
plt.title("Mega-Watt Hours v Temperature and Humidex")
plt.grid()

plt.show()

# Fit a regression model

mdl_TH = sm.OLS.from_formula('MWh ~ 1 + Temperature + Humidex', \
                             power_consumption)

results_TH = mdl_TH.fit()

print(results_TH.summary())


# Overlay the fitted model

# Sort the data so that MWh is always increasing
df_plot = power_consumption.sort_values("MWh")

# Just get the columns of interest
df_plot = df_plot.loc[:,["Temperature","Humidex"]]

# reset the index
df_plot.reset_index(drop=True,inplace=True)

# Get predicted values
yhat = results_TH.predict(df_plot)

# Plot the original data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(power_consumption["Temperature"], \
           power_consumption["Humidex"], \
           power_consumption["MWh"],label="raw data")

plt.xlabel("Temperature")
plt.ylabel("Humidex")
ax.set_zlabel("Mega-Watt Hours")
plt.title("Mega-Watt Hours v Temperature and Humidex")
plt.grid()

# Overlay the model fit
ax.plot(df_plot["Temperature"],df_plot["Humidex"],yhat,color="r",\
        label="fitted model")

plt.legend()
plt.show()

# Calculate the sample variance
Y = power_consumption["MWh"]
yhat = results_TH.predict(power_consumption)
resids = (yhat-Y)
N = len(Y)
k = len(results_TH.params)
dof = N - k
s2 = np.sum(resids**2)/dof

print("The sample variance is {val:.6f}".format(val=s2))

# Calculate the variance matrix
N = len(power_consumption)
X = np.ones((N,3))
X[:,1:] = power_consumption.loc[:,["Temperature","Humidex"]]

V = np.linalg.inv(np.matmul(X.transpose(),X))

print("The variance matrix is:")
print(V)


# Calculate the t-values
beta = results_TH.params
ti = beta/np.sqrt(np.diag(V))/np.sqrt(s2)

print("The t-value for the constant is {val:.6f}".format(val=ti[0]))
print("The t-value for the Temperature is {val:.6f}".format(val=ti[1]))
print("The t-value for the Humidex is {val:.6f}".format(val=ti[2]))

# These can be obtained from the model fit using
results_TH.tvalues

# Calulate the critical value
tc = t.ppf(0.025, dof) 

print("The critical value for the t-value is {val:.6f}".format(val=tc))

####################################################################
# Comparing models
####################################################################

# Generate a constant model
model_0 = sm.OLS.from_formula('MWh ~ 1', power_consumption)
results_0 = model_0.fit()
print(results_0.summary())

# Calculate the F value comparing with the constant model
m = 2;
F0 = ((results_TH.rsquared)-(results_0.rsquared))\
    /m/(1-(results_TH.rsquared))*(N-k)
F0c = f.ppf(0.95,m,N-k)
pval0 = f.cdf(1/F0,N-k,m)

print("")
print("The F-test value for comparing to the constant model is {val:.6f}"\
      .format(val=F0))
print("The critical value for the F-test is {val:.6f}".format(val=F0c))
print("The p-value for the F-test is {val:.6f}".format(val=pval0))

# These can be obtained from the model fit using
[results_TH.fvalue,results_TH.f_pvalue]

#### Calculate the F value comparing TH Model with the Temperature model###
m = 1;
FT = ((results_TH.rsquared)-(results_T.rsquared))\
    /m/(1-(results_TH.rsquared))*(N-k)
FTc = f.ppf(0.95,m,N-k)
pvalT = f.cdf(1/FT,N-k,m)

print("")
print("The F-test value for comparing to the Temperature model is {val:.6f}"\
      .format(val=FT))
print("The critical value for the F-test is {val:.6f}".format(val=FTc))
print("The p-value for the F-test is {val:.6f}".format(val=pvalT))

# These are not calculated for the model fit, but can be 
# using the f_test method.

#### Calculate the F value comparing TH Model with the Temperature model###
m = 1;
FT = ((results_TH.rsquared)-(results_H.rsquared))\
    /m/(1-(results_TH.rsquared))*(N-k)
FTc = f.ppf(0.95,m,N-k)
pvalT = f.cdf(1/FT,N-k,m)

print("")
print("The F-test value for comparing to the Humitidex model is {val:.6f}"\
      .format(val=FT))
print("The critical value for the F-test is {val:.6f}".format(val=FTc))
print("The p-value for the F-test is {val:.6f}".format(val=pvalT))

