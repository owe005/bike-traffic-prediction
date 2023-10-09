from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import holidays # Feature engineering for holidays
import numpy as np

dir = 'ready_data/ready_data.csv'

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

# Define X as all weather columns information and y as Trafikkmengde
X = df.drop(columns=['Datotid', 'Trafikkmengde'])
y = df['Trafikkmengde']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=1)

# Initialize the model and train it
rf = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=1) # Just because I want it to start quickly, I have lowered the params, accuracy still high

rf.fit(X_train, y_train)

# Predict on val set
rf_predict = rf.predict(X_val)

# RMSE
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_predict))

print('RMSE: ', rf_rmse)
