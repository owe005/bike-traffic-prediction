from flask import Flask, render_template, request
from model import rf, norway_holidays, X_train, df # Importing our model and dataframe
import numpy as np
import pandas as pd

# In this script we've already determined the best hyperparameters for RandomForestRegressor using GridSearchCV which was done in model.ipynb

app = Flask(__name__)

# Function to get for empty values field, will take median for the field if left empty.
def get_or_default(form, field, default_value):
    value = form.get(field)
    return float(value) if value else default_value

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Getting data from form
        date_time = request.form['date-time']
        date_time = pd.to_datetime(date_time)

        # Extracting data from datetime
        month = date_time.month
        day_of_week = date_time.dayofweek
        hour = date_time.hour
        is_holiday = date_time.date() in norway_holidays
        is_weekend = day_of_week >= 5
        is_rushhour = hour in [7, 8, 15, 16, 17]
        is_night = hour in [0, 1, 2, 3, 4, 5]
        summer = month in [6, 7, 8]
        winter = month in [12, 1, 2]
        spring = month in [3, 4, 5]
        autumn = month in [9, 10, 11]

        # Running our function on every field, checking for if the user has inputted something, else median
        globalstraling = get_or_default(request.form, 'globalstraling', df['Globalstraling'].median())
        solskinstid = get_or_default(request.form, 'solskinstid', df['Solskinstid'].median())
        lufttemperatur = get_or_default(request.form, 'lufttemperatur', df['Lufttemperatur'].median())
        vindsstyrke = get_or_default(request.form, 'vindstyrke', df['Vindstyrke'].median())
        lufttrykk = get_or_default(request.form, 'lufttrykk', df['Lufttrykk'].median())
        vindkast = get_or_default(request.form, 'vindkast', df['Vindkast'].median())

        # Preparing information, turning into a dataframe and feeding it to model
        input_data = np.array([[globalstraling, solskinstid, lufttemperatur, vindsstyrke, lufttrykk, vindkast, month, day_of_week, hour, is_holiday, is_weekend, is_rushhour, is_night, summer, winter, spring, autumn]])        
        input_data = pd.DataFrame(input_data, columns=X_train.columns)

        print(input_data)

        # Predict, round by 2 decimal points
        prediction = round(rf.predict(input_data)[0], 2)

    return render_template('index.html', prediction=prediction)    

if __name__ == '__main__':
    app.run(port=8080, debug=True)