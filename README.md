# Disaster Response Pipeline Project

### Instructions:
In order to increase my prospects and potential as a data scientist, I have studied and developed my data engineering abilities in this project. In order to create a model for an API that categorizes disaster messages, I'll use these abilities in this project to evaluate disaster data from Appen.


You developed a machine learning pipeline to classify these events so I could communicate the information to the proper disaster relief organization.
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
