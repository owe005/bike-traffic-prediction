"""
INF161project.py apart of the deliverables, this file contains all my code used to reproduce my work.
Because all my code essentially comes from different files, I will try to organize it for easier viewing.

I will include comments where I deem it necessary, but methodological choices, expectations and results will be in the INF161report.pdf file.

Author: Ole Kristian Westby, owe009@uib.no | INF161
"""


### Part 01, preparation.

from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import ElasticNet
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import seaborn as sns
import os
import holidays # Feature engineering for holidays
import matplotlib.pyplot as plt

dir_weather = "data/raw_data/weather_data/"

# List of all .csv files in /weather_data/
files = [f for f in os.listdir(dir_weather) if f.endswith('.csv')]

# Interesting columns
columns = ["Dato", "Tid", "Globalstraling", "Solskinstid", "Lufttemperatur", "Vindstyrke", "Lufttrykk", "Vindkast"] # Added "lufttrykk" later because it had a big difference

# Individual dataframes, initialized as empty list
dfs = []

# For every .csv file, append to dataframe list
for file in files:
    file_path = os.path.join(dir_weather, file)
    df = pd.read_csv(file_path, usecols=columns)
    dfs.append(df)

# Merge all dataframes into one
merged_weather_df = pd.concat(dfs, ignore_index=True)

# Convert to pandas datetime format, and filter for 2015 + later
merged_weather_df["Dato"] = pd.to_datetime(merged_weather_df["Dato"])
merged_weather_df = merged_weather_df[merged_weather_df["Dato"].dt.year >= 2015]

# Counting how many missing-data-rows
rows_missing_data = merged_weather_df[merged_weather_df.isna().any(axis=1)].shape[0]

# Replace the missing data rows, conver to dtypes, drop missing values
merged_weather_df.replace(9999.99, np.nan, inplace=True)
merged_weather_df = merged_weather_df.convert_dtypes()
merged_weather_df.dropna(inplace=True)

# Converting both to a string to combine them
merged_weather_df["Dato"] = merged_weather_df["Dato"].astype(str)
merged_weather_df["Tid"] = merged_weather_df["Tid"].astype(str)

# Combining, to datetime, drop old cols
merged_weather_df["Datotid"] = merged_weather_df["Dato"] + " " + merged_weather_df["Tid"]
merged_weather_df["Datotid"] = pd.to_datetime(merged_weather_df["Datotid"])
merged_weather_df.drop(["Dato", "Tid"], axis=1, inplace=True)

# Reset index
merged_weather_df = merged_weather_df.reset_index(drop=True)

# Initialize traffic data as dataframe
dir_traffic = "data/raw_data/traffic_data/trafikkdata.csv"
traffic_df = pd.read_csv(dir_traffic, delimiter=";")

# Filter dataframe where "Felt" == "Totalt" and keep interesting cols
traffic_df = (
    traffic_df[traffic_df["Felt"] == "Totalt"]
    .loc[:, ["Dato", "Fra tidspunkt", "Trafikkmengde"]]

    # New datetime format and combining cols
    .assign(Datotid=lambda x: pd.to_datetime(x["Dato"] + " " + x["Fra tidspunkt"]))
    
    # Drop old cols
    .drop(columns=["Dato", "Fra tidspunkt"])
)

# Filter for missing data, fix indexing and set index to datetime
traffic_df.dropna(inplace=True)
traffic_df = traffic_df.reset_index(drop=True)
traffic_df.set_index("Datotid", inplace=True)

merged_weather_df.set_index("Datotid", inplace=True)

# Perfect opportunity to rename the variable.
weather_df = merged_weather_df.resample("H").mean()

# Join to one dataframe
ready_df = traffic_df.join(weather_df, how="inner")

# Setting traffic volume to an int as it was previously an object
ready_df["Trafikkmengde"] = pd.to_numeric(ready_df["Trafikkmengde"], errors="coerce")
ready_df["Trafikkmengde"].fillna(ready_df["Trafikkmengde"].mean(), inplace=True)
ready_df["Trafikkmengde"] = ready_df["Trafikkmengde"].astype(int)

ready_df.reset_index(inplace=True)

# At this point we're left with 65361 rows of data.

## Data exploration

# Best time
best_hour = ready_df.groupby(ready_df["Datotid"].dt.hour)["Trafikkmengde"].mean().idxmax()
print(f"Best time of day is: {best_hour}:00\n----------------------------------------------------")

# Average per day
daily_cyclists_per_day = ready_df.resample("D", on="Datotid")["Trafikkmengde"].sum()
avg_cyclists_per_day = daily_cyclists_per_day.groupby(daily_cyclists_per_day.index.dayofweek).mean()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_cyclists_per_day.index = avg_cyclists_per_day.index.map(lambda x: days[x])
print(f"The average cyclists per day: \n {avg_cyclists_per_day} \n ----------------------------------------------------")

# Best temperature
best_temp = ready_df.groupby('Lufttemperatur')['Trafikkmengde'].mean().idxmax()
print(f"The temperature with the highest average bicycle traffic is: {round(best_temp,1)}°C")

# Best windstrength
best_wind = ready_df.groupby('Vindstyrke')['Trafikkmengde'].mean().idxmax()
print(f"The wind strength with the highest average bicycle traffic is: {round(best_wind, 2)} m/s")

ready_dir = "data/ready_data/ready_data.csv"
ready_df.to_csv(ready_dir, index=False)

# Drop rows with NaN values, I had to do this for visualizations.
clean_ready_df = ready_df.dropna(subset=['Vindstyrke', 'Trafikkmengde'])

## Visualization

figures_dir = 'figures/'

## Wind speed vs. Traffic (These are saved in /figures/)
# Setting up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(clean_ready_df['Vindstyrke'], clean_ready_df['Trafikkmengde'], s=3)
ax.set_title('Wind Speed vs. Traffic')
ax.set_xlabel('Wind Speed')
ax.set_ylabel('Traffic Volume')
plt.savefig(figures_dir + 'windspeedvstraffic.png')

## Temperature vs. Traffic
# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(clean_ready_df['Lufttemperatur'], clean_ready_df['Trafikkmengde'], s=3)
ax.set_title('Temperature vs. Traffic')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Traffic Volume')
plt.savefig(figures_dir + 'tempvstraffic.png')

## Traffic vs. Weekdays

clean_ready_df = clean_ready_df.copy() # Got a warning, decided to go with it.

# Get names from column
clean_ready_df['Weekday'] = clean_ready_df['Datotid'].dt.day_name()

# Get date from column
clean_ready_df['Date'] = clean_ready_df['Datotid'].dt.date

# Calculate daily traffic
daily_traffic = clean_ready_df.groupby(['Date', 'Weekday'])['Trafikkmengde'].sum().reset_index()

# Average daily traffic for each weekday
avg_daily_traffic_per_weekday = daily_traffic.groupby('Weekday')['Trafikkmengde'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).sort_values()

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
avg_daily_traffic_per_weekday.plot(kind='bar', ax=ax)
ax.set_title('Traffic vs Weekday')
ax.set_xlabel('Weekday')
ax.set_ylabel('Average Daily Traffic Volume')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.savefig(figures_dir + 'trafficvsweekdays.png')

## Lufttrykk vs. Trafikkmengde
plt.figure(figsize=(10, 6))
plt.scatter(clean_ready_df['Lufttrykk'], clean_ready_df['Trafikkmengde'], alpha=0.5)
plt.title('Lufttrykk vs. Trafikkmengde')
plt.xlabel('Lufttrykk')
plt.ylabel('Trafikkmengde')
plt.grid(True)
plt.savefig(figures_dir + 'lufttrykkvstraffic.png')

## Vindkast vs. Trafikkmengde
plt.figure(figsize=(10, 6))
plt.scatter(clean_ready_df['Vindkast'], clean_ready_df['Trafikkmengde'], alpha=0.5)
plt.title('Vindkast vs. Trafikkmengde')
plt.xlabel('Vindkast')
plt.ylabel('Trafikkmengde')
plt.grid(True)
plt.savefig(figures_dir + 'vindkastvstraffic.png')

## 2017 traffic (looking at SykkelVM)
df_2017 = clean_ready_df[clean_ready_df['Datotid'].dt.year == 2017]
daily_traffic_2017 = df_2017.resample('D', on='Datotid')['Trafikkmengde'].sum()
plt.figure(figsize=(15, 7))
daily_traffic_2017.plot()
plt.title('Daily Traffic for 2017')
plt.xlabel('Date')
plt.ylabel('Trafikkmengde')
plt.grid(True)
plt.tight_layout()
plt.savefig(figures_dir + 'dailytraffic2017.png')

## Correlation matrix
correlation = ready_df.corr()
correlation = ready_df.drop(columns=['Datotid']).corr()
numeric_df = ready_df.select_dtypes(include=['float64', 'int64'])
correlation = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.savefig(figures_dir + 'correlationmatrix.png')

### Part 02, data modelling

dir = 'data/ready_data/ready_data.csv'

# Load data into a dataframe
df = pd.read_csv(dir)

missing_data = df.isnull().sum()

# Impute with median
for col in missing_data.index[missing_data > 0]:
    df[col].fillna(df[col].median(), inplace=True)

df['Datotid'] = pd.to_datetime(df['Datotid'])

# Extracting time-related stuff
df['Month'] = df['Datotid'].dt.month
df['DayOfWeek'] = df['Datotid'].dt.dayofweek # Monday: 0, Tuesday: 1, ... , Sunday: 6.
df['Hour'] = df['Datotid'].dt.hour

# Feature engineering: Public holidays in Norway
norway_holidays = holidays.Norway()
df['IsHoliday'] = df['Datotid'].apply(lambda x: pd.to_datetime(x).date() in norway_holidays)

# Feature engineering: Weekends, rushhour. I tested different options on "rushhour" and found this to be the best
df['IsWeekend'] = df['Datotid'].dt.dayofweek >= 5
df['IsRushhour'] = df['Hour'].isin([7, 8, 15, 16, 17])

df['IsNight'] = df['Hour'].isin([0, 1, 2, 3, 4, 5])

# Feature engineering: Seasons
df['Summer'] = df['Month'].isin([6, 7, 8])
df['Winter'] = df['Month'].isin([12, 1, 2])
df['Spring'] = df['Month'].isin([3, 4, 5])
df['Autumn'] = df['Month'].isin([9, 10, 11])

df.reset_index(drop=True, inplace=True)

# X: all weather information, y: only trafikkmengde.
X = df.drop(columns=['Datotid', 'Trafikkmengde'])
y = df['Trafikkmengde']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Split the training+validation data again to get separate training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)


# 32680 training samples, 32681 validation samples

'''
## LinearRegression

# Initialize the model and train it
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on the validation set
lr_predict = lr.predict(X_val)

# Calulate RMSE (root of mean squared error)
lr_rmse = np.sqrt(mean_squared_error(y_val, lr_predict)) # ~61,7 RMSE
'''

'''
## Lasso

# Initialize the model and train it
lasso = Lasso(alpha=0.1, random_state=1)
lasso.fit(X_train, y_train)

# Predict on val set
lasso_predict = lasso.predict(X_val)

# RMSE
lasso_rmse = np.sqrt(mean_squared_error(y_val, lasso_predict)) # ~61,7 RMSE
'''

'''
## GradientBoostingRegressor

# Initialize the model and train it
gb = GradientBoostingRegressor(n_estimators=10, random_state=1)
gb.fit(X_train, y_train)

# Predict on val set
gb_predict = gb.predict(X_val)

# RMSE
gb_rmse = np.sqrt(mean_squared_error(y_val, gb_predict)) # ~53,1 RMSE
'''
'''
## ElasticNet

# Initialize the model and train it
elastic_net = ElasticNet(alpha=1, random_state=1)
elastic_net.fit(X_train, y_train)

# Predict on val set
y_predict = elastic_net.predict(X_val)

# RMSE
en_rmse = np.sqrt(mean_squared_error(y_val, y_predict)) # ~61,8 RMSE
'''

'''
## KNeighborsRegressor

# Initialize the model and train it
kn = KNeighborsRegressor(n_neighbors=10)
kn.fit(X_train, y_train)

# Predict on val set
y_predict = kn.predict(X_val)

# RMSE
kn_rmse = np.sqrt(mean_squared_error(y_val, y_predict)) # ~49,7 RMSE
'''


## RandomForestRegressor

# Initialize the model and train it
rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=1) # Tune the n_estimators parameter for it to run faster. 

rf.fit(X_train, y_train)

# Predict on val set
rf_predict = rf.predict(X_val)

# RMSE
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_predict)) # ~22.65 RMSE

## Model found, now GridSearchCV.

"""

# GridSearch lets us check for the best hyperparameters with a given parameter grid

# NB: This part has been commented out simply because it takes ~35 minutes to run, it will find the best parameters.

# Define parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Let's GridSearch RandomForest with this param grid
grid_search = GridSearchCV(estimator=rf, 
                           param_grid=param_grid, 
                           cv=3, 
                           n_jobs=-1, 
                           verbose=2, 
                           scoring='neg_mean_squared_error') # MSE is positive or zero, neg is prefix to all mse

# We will attempt to fit all different parameters on the training data.
grid_search.fit(X_train, y_train)

# Best params
best_params = grid_search.best_params_

# Best estimator
best_rf = grid_search.best_estimator_

# Predict on val data
rf_predict = best_rf.predict(X_val)

# RMSE
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_predict))

# Print results
print(rf_rmse)
print(best_rf)

"""

### Part 03, prediction

# Dir is for dataset in
dir = "data/raw_data/weather_data/Florida_2023-01-01_2023-07-01_1688719120.csv"
# Out is for predictions out
out = "predictions.csv"

df = pd.read_csv(dir)

# Change dato and tid to a datetime object
df['Datotid'] = pd.to_datetime(df['Dato'] + ' ' + df['Tid'])

# Drop old cols and Datotid leftmost
df.drop(['Dato', 'Tid'], axis=1, inplace=True)
cols = ['Datotid'] + [col for col in df if col != 'Datotid']
df = df[cols]

# Define missing data, replace with NaN
df.replace(9999.99, np.nan, inplace=True)

# Fill all missing data with median
for col in df.columns:
    df[col].fillna(df[col].median(), inplace=True)

# Columns we will feed into model for predictions
interesting_cols = ['Datotid', 'Globalstraling', 'Solskinstid', 'Lufttemperatur', 'Vindstyrke', "Lufttrykk", 'Vindkast']
X_predict = df[interesting_cols]

# Extract month, dayofweek and hour from datotid col
X_predict['Month'] = X_predict['Datotid'].dt.month
X_predict['DayOfWeek'] = X_predict['Datotid'].dt.dayofweek
X_predict['Hour'] = X_predict['Datotid'].dt.hour

# Feature engineering: Public holidays in Norway
norway_holidays = holidays.Norway()
X_predict['IsHoliday'] = X_predict['Datotid'].apply(lambda x: pd.to_datetime(x).date() in norway_holidays)

# Feature engineering: Weekends, rushhour. I tested different options on "rushhour" and found this to be the best
X_predict['IsWeekend'] = X_predict['Datotid'].dt.dayofweek >= 5
X_predict['IsRushhour'] = X_predict['Hour'].isin([7, 8, 15, 16, 17])

X_predict['IsNight'] = X_predict['Hour'].isin([0, 1, 2, 3, 4, 5])

# Feature engineering: Seasons
X_predict['Summer'] = X_predict['Month'].isin([6, 7, 8])
X_predict['Winter'] = X_predict['Month'].isin([12, 1, 2])
X_predict['Spring'] = X_predict['Month'].isin([3, 4, 5])
X_predict['Autumn'] = X_predict['Month'].isin([9, 10, 11])

# Drop datotid
X_predict.drop(['Datotid'], axis=1, inplace=True)

# Predict
predicted_traffic = rf.predict(X_predict)

# New col: predicted traffic
df['Prediksjon'] = predicted_traffic

# Retain only datotid and prediksjon
df = df[['Datotid', 'Prediksjon']]

# Revert datotid to two cols, dato and tid.
df['Dato'] = df['Datotid'].dt.date
df['Tid'] = df['Datotid'].dt.time

# Drop old datotid col
df.drop(['Datotid'], axis=1, inplace=True)

# Reorder cols
df = df[['Dato', 'Tid', 'Prediksjon']]

# Save to CSV
df.to_csv(out, index=False)

print("Predictions ready, see predictions.csv for predictions.")