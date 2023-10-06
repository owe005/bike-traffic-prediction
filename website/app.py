from flask import Flask, render_template, request
from model import * # Importing our model and dataframe
import numpy as np
import pandas as pd

# In this script we've already determined the best hyperparameters for RandomForestRegressor using GridSearchCV which was done in model.ipynb

app = Flask(__name__)

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

        globalstraling = get_or_default(request.form, 'globalstraling', df['Globalstraling'].median())
        solskinstid = get_or_default(request.form, 'solskinstid', df['Solskinstid'].median())
        lufttemperatur = get_or_default(request.form, 'lufttemperatur', df['Lufttemperatur'].median())
        vindsstyrke = get_or_default(request.form, 'vindstyrke', df['Vindstyrke'].median())
        vindkast = get_or_default(request.form, 'vindkast', df['Vindkast'].median())

        input_data = np.array([[globalstraling, solskinstid, lufttemperatur, vindsstyrke, vindkast, month, day_of_week, hour]])
        input_data = pd.DataFrame(input_data, columns=X_train.columns)

        print(input_data)

        prediction = rf.predict(input_data)[0]
            

    return render_template('index.html', prediction=prediction)    

if __name__ == '__main__':
    app.run(port=8080, debug=True)