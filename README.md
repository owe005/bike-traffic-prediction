# **INF161 - Bike Traffic Prediction**
#### *Ole Kristian Westby | owe009@uib.no | H23*
This is apart of my project for the INF161 Datascience course I am taking autumn semester 2023. From the assignment description:

 *"The goal of this project is to apply all material learned in INF161 and successfully complete a data
science project"*

## This project is split into 4 parts:

**Data preparation and exploratory data analysis**

**Modelling and prediction**

**Prediction**

**Website**

## File explanations:

- preparation.ipynb: contaions all the data exploration, analysis and preparation steps needed before training the model.
- model.ipynb: detailed process of model training, hyperparameter tuning, and model selection.
- website.zip: a website version of this project, which allows a user to input a date, time and optional weather data to get a prediction of amount of cyclist at that time.
- data/ready_data/*: datasets prepared for model training after preparation.ipynb steps.
- data/raw_data/*: raw datasets, weather and traffic data.
- INF161project.py: final deliverable containing all code
- INF161report.pdf: final deliverable containing project report
- figures/*: data exploration, analysis figures

## Getting Started
To use the website:
1. Unpack website.zip into a directory and run ```python app.py```. This will take a second. (The amount of n_estimators direct this)
2. Once ready, visit localhost:8080/
3. Input desired details and optional weather data, and click ```predict```.

To use the model:
1. Make sure you have a .csv file with the same columns like the ones in raw data and/or ready_data. 
2. Dir is for dataset in, dir = "data/raw_data/weather_data/Florida_2023-01-01_2023-07-01_1688719120.csv" Out is for predictions out out = "predictions.csv"
3. Running INF161project.py will predict on the weather data set and put all the predictions in a predictions.csv file in the root
###
