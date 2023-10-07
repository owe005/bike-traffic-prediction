# **INF161 - Bike Traffic Prediction**
#### *Ole Kristian Westby | owe009@uib.no | H23*

This is apart of my project for the INF161 Datascience course I am taking autumn semester 2023. From the assignment description:

 *"The goal of this project is to apply all material learned in INF161 and successfully complete a data
science project"*

<img src= "webpage.png" width="300">

## This project is split into 4 parts:

**Data preparation and exploratory data analysis**

**Modelling and prediction**

**Prediction**

**Website**

## File explanations:

- preparation.ipynb: contaions all the data exploration, analysis and preparation steps needed before training the model.
- model.ipynb: detailed process of model training, hyperparameter tuning, and model selection.
- prediction.py: A script to provide the model with 2023 weather data from 01-01-2023 to 01-07-2023 and predict the bicycle traffic for that year, given the weather data.
- model.py: contains the model, a more compact version of model.ipynb that only has the important bits needed to predict. Prediction.py utilized this to make predictions.
- website/*: a website version of this project, which allows a user to input a date, time and optional weather data to get a prediction of amount of cyclist at that time.
- website/model.py: a replica of the trained model for web deployment.
- website/app.py: flask server
- website/templates/*: html files for web interface
- website/static/*: css styling
- ready_data/*: datasets prepared for model training after preparation.ipynb steps.
- raw_data/*: raw datasets, weather and traffic data.
- predictions/*: weather data for 2023 and the predicted traffic, all in predictions.csv.
- deliverables/*: submission materials

## Getting Started
To use the website:
1. Navigate to /website/ directory and run ```python app.py```. This will take a second.
2. Once ready, visit localhost:8080/
3. Input desired details and optional weather data, and click ```predict```.

To use the model:
1. Make sure you have a .csv file with the same columns like the ones in raw data and/or ready_data. 
2. Then all you have to do is change the variables ```dir```
and ```out``` on line 5, 6 in predictions.py. Then run python predictions.py.
3. Make sure that you are within the working directory of predictions.py and that the dataset you have is in the /predictions/ folder.
###