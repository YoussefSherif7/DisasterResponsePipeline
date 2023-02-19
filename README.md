# Disaster Response Pipeline Project


### Table of contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
This project is built using python and a few libraries. 
The libraries used are: (Pandas, Numpy, Sqlalchemy, sklearn, nltk)

## Project Motivation<a name="motivation"></a>
This is my second project in the Udacity Data Science program.
The idea behind this project is using a data set containing real messages that were sent during disaster events, and create a machine learning pipeline in order to categorize messages to send them to the correct disaster relief support.
The project is divided into 3 sections.
### 1. ETL (Extract, Transform, load)
This pipeline is responsible for extracting the data from the provided dataset, followed by a transformation/cleaning process of merging/removing duplicates etc.

### 2. ML pipeline
This is the machine learning pipeline where the data is normalized and tokenized, and then builds the machine learning pipelines that trains on the data and produces a trained model for future predictions.

### 3. Webapp 
The final section is the way the model is displayed. Uing a webapp, after linking the model and table, inputing a message will categorize the message to its appropriate disaster response category and highlights them in a table.

##  File Descriptions <a name="files"></a>
The repository contains 3 folders.
1. The app folder handles the webapp and contains the web templates as well as the script that runs the webapp.
2. The data folder which contains the 2 csv files provided, the database created named "DisasterResponse.db", and finally the process_data.py script which begins the ETL pipeline process.
3. The models folder which contains the model produced as a pickle file, and the train_classifier.py script which is responsible for the ML pipeline building. 
4. This ReadME file.

## Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results<a name="results"></a>
The output of the entire project is a trained model linked to a webapp that classifies messages sent after disasters.
An example of the output working can be seen in the screenshots below, where a message is written in the text bar and the corresponding categories are highlighted.
![Screenshot 2023-02-20 010911](https://user-images.githubusercontent.com/71082811/219980891-744ecc4d-1021-4ccb-a85b-805aa2979dec.png)
![Screenshot 2023-02-20 010858](https://user-images.githubusercontent.com/71082811/219980892-9716b2d0-a568-4107-9f6d-462b4f8677dc.png)
![Screenshot 2023-02-20 010833](https://user-images.githubusercontent.com/71082811/219980893-87424ff8-5a7d-4012-8c86-17c246ca47e4.png)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Thanks to the Udacity data science team for their support throughout this program so far, and Appen (formerly figure eight) for the data sets provided.
