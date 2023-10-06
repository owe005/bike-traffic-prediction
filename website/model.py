from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

dir = '../ready_data/ready_data.csv'

# Load data into a dataframe
df = pd.read_csv(dir)

# Check for missing data
missing_data = df.isnull().sum()

# Impute missing data with median
for col in missing_data.index[missing_data > 0]:
    df[col].fillna(df[col].median(), inplace=True)

# Datetime format
df['Datotid'] = pd.to_datetime(df['Datotid'])

# Extracting time-related stuff
df['Month'] = df['Datotid'].dt.month
df['DayOfWeek'] = df['Datotid'].dt.dayofweek # Monday: 0, Tuesday: 1, ... , Sunday: 6.
df['Hour'] = df['Datotid'].dt.hour

# Define X as all weather columns information and y as Trafikkmengde
X = df.drop(columns=['Datotid', 'Trafikkmengde'])
y = df['Trafikkmengde']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=1)

# Initialize the model and train it
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=4, random_state=1) # With 100 estimators, the RMSE is 26.1455 instead, but it took a minute to run. # 1000: 25,89

rf.fit(X_train, y_train)

# Predict on val set
rf_predict = rf.predict(X_val)

# RMSE
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_predict))