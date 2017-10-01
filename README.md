# Recursive-Forecasting-using-Multi-Output-Regression-with-Statistical-Analysis
This project aims to forecast the Median Selling Price (MSP) of houses in MA and NY states for 2 months based on 70 months of data using time series analysis in Python by using Multi Output Regressor on three traditional regression models Lasso, KNN and Gradient Boosting Regressor. Statistical analysis is also performed to understand the influence of different features on the target label (MSP). The data is collected from Zillow website.

All the data files are in the _Data_ folder and the _Submission_ folder has _Code_ and _Figures_ folder.

# Problem #
Forecast the Median Selling Price per square feet for 2 months in 14 steps, 7 steps in each month for two states MA and NY.

# How to Run #
Go to _Submission/Code_ folder and run _MA_run_me.py_ and _NY_run_me.py_ python scripts using the following commands: `python MA_run_me.py` and `NY_run_me.py`. These can be run in two terminals simultaneously.
   
# Dataset Description #
1. Each dataset ('MA_dataset.csv' and NY_dataset.csv') has 490 data cases with each data case having 10 features (described below) and the target value as Median Selling Price per sqft. (MSP).
2. There are 7 values in each month which are dated as 5, 10, 15, 18, 20, 25 and 28. The data is for 70 months which range from October 2010 to July 2016.

# Feature Information #
The features that are considered in the raw dataset are the following:

1. MLP ($) - Median List Pricing per sqft.
   1. Description - Median of the list price (or asking price) for homes listed on Zillow divided by the square footage of a home.
   2. Type - continuous real float64 value
    
2. MPC (%) - Median Price Cut
   1. Description - Median of the percentage price reduction for homes with a price reduction during the month.
   2. Type - continuous real float64 value between 0 and 100

3. SFL (%) - Sold For Loss
   1. Description - The percentage of homes in an area that sold for a price lower than the previous sale price.
   2. Type - continuous real float64 value between 0 and 100
    
4. SFG (%) - Sold For Gain
   1. Description - The percentage of homes in an area that sold for a price higher than the previous sale price.
   2. Type - continuous real float64 value between 0 and 100
    
5. IV (%) - Increasing Values
   1. Description - The percentage of homes in an given region with values that have increased in the past year.
   2. Type - continuous real float64 value between 0 and 100
    
6. DV (%) - Decreasing Values
   1. Description - The percentage of homes in an given region with values that have decreased in the past year.
   2. Type - continuous real float64 value between 0 and 100
    
7. TNV (%) - Turnover
   1. Description - The percentage of all homes in a given area that sold in the past 12 months.
   2. Type - continuous real float64 value between 0 and 100
    
8. BSI - Buyer Seller Index
   1. Description - This index combines the sale-to-list price ratio, the percent of homes that subject to a price cut and the time properties spend on the market (measured as Days on Zillow). Higher numbers indicate a better buyers’ market, lower numbers indicate a better sellers’ market, relative to other markets within a metro.
   2. Type - continuous real float64 value

9. PTR - Price To Rent Ratio
   1. Description - This ratio is first calculated at the individual home level, where the estimated home value is divided by 12 times its estimated monthly rent price. The the median of all home-level price-to-rent ratios for a given region is then calculated.
   2. Type - continuous real float64 value
    
10. MHI - Market Health Index
   1. Description - This index indicates the current health of a given region’s housing market relative to other markets nationwide. It is calculated on a scale from 0 to 10, with 0 representing the least healthy markets and 10 the healthiest markets.
   2. Type - continuous real float64 value
    
# Target Label MSP #
MSP - Median Selling Price
   1. Description - Median of the selling price for all homes sold in a given region divided by the square footage of a home.
   2. Type - continuous real float64 value
    
    
# Lag Features #
1. To do a time series forecast in which we don't have features for that period of time to predict for that same period, we need to consider lags, meaning the features for the current period will be the features and the target label from the previous n number of periods and this is called a lag. The optimal number of lags is determined and a new csv file 'MA_Lagged_data.csv' (same with NY) is formed with all the features lagged according to the optimal number of lags.
2. We got an optimal number of 8 lags. So, the 'MA_Lagged_data.csv' file has 481 data cases with 88 features (11*8) and two target labels, so totaling to 90 columns. We are performing multi output regression with two target labels, one being the current prediction and the other is the next step's prediction. As the number of lags is 8, the first 8 datacases from 'MA_dataset.csv' and the last one datacase are chopped as they contain NaN values because of shifting and hence, the difference between number of datacases between 'MA_dataset.csv' and 'MA_Lagged_data.csv'.
3. PCA is performed to find the 14 best features and a new csv is formed with the optimal features 'MA_New_Lagged_data.csv'. So, this file has 481 data cases with 14 features.


# Results #
The forecasted labels, actual values and RMSE are printed on the console after the run finishes. The average RMSE for MA state is found as 35.2 whereas for NY state, it is 25.76.
