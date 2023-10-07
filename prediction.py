import pandas as pd
import numpy as np
from model import rf

dir = "predictions/Florida_2023-01-01_2023-07-01_1688719120.csv"
out = "predictions/predictions.csv"

df = pd.read_csv(dir)

df['Datotid'] = pd.to_datetime(df['Dato'] + ' ' + df['Tid'])
df.drop(['Dato', 'Tid'], axis=1, inplace=True)

cols = ['Datotid'] + [col for col in df if col != 'Datotid']
df = df[cols]

missing_indicator = 9999.99
df.replace(missing_indicator, np.nan, inplace=True)

for col in df.columns:
    df[col].fillna(df[col].median(), inplace=True)

# Columns we will feed into model for predictions
interesting_cols = ['Datotid', 'Globalstraling', 'Solskinstid', 'Lufttemperatur', 'Vindstyrke', 'Vindkast']

X_predict = df[interesting_cols]

X_predict['Month'] = X_predict['Datotid'].dt.month
X_predict['DayOfWeek'] = X_predict['Datotid'].dt.dayofweek
X_predict['Hour'] = X_predict['Datotid'].dt.hour
X_predict.drop(['Datotid'], axis=1, inplace=True)

predicted_traffic = rf.predict(X_predict)
df['PREDICTED TRAFFIC'] = predicted_traffic

df.to_csv(out, index=False)

print("Predictions ready, see /predictions/predictions.csv for predictions.")