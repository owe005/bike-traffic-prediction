import pandas as pd
import numpy as np
from model import rf
import holidays # Feature engineering for holidays

# Dir is for dataset in
dir = "raw_data/weather_data/Florida_2023-01-01_2023-07-01_1688719120.csv"
# Out is for predictions out
out = "predictions/predictions.csv"

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

# Feature engineering: Seasons
X_predict['Summer'] = X_predict['Month'].isin([6, 7, 8])
X_predict['Winter'] = X_predict['Month'].isin([12, 1, 2])
X_predict['Spring'] = X_predict['Month'].isin([3, 4, 5])
X_predict['Autumn'] = X_predict['Month'].isin([9, 10, 11])

X_predict['IsNight'] = X_predict['Hour'].isin([0, 1, 2, 3, 4, 5])

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

print("Predictions ready, see /predictions/predictions.csv for predictions.")