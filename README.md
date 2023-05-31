# Udacity Datascience Nanodegree Project 2: Disaster Response Pipelines

__Project description:__ The second project due as part of Udacity's Data Science Nanodegree involves building a web app that can classify messages to show how they relate to natural disasters.

There are three stages to this project:
1) Extracting and cleaning data
2) Using a machine learning pipeline to create a model for classifying data
3) Deploying the model as a web app using Flask.

The end result is an app that visualises the data and uses the model developed in stage 2 to classify messages.

__File descriptions:__ This project contains several files. The first two files are jupyter notebooks that work through the process of building the ETL and machine learning pipeline, while offering some narrative explaining any difficulties faced and decisions made. These file are:

ETL Pipeline Preparation.ipynb (LINK) - This outlines how the ETL pipeline is made and presents functions that can be used in the web app 
ML Pipeline Preparation.ipynb (LINK) - This file does the same for the machine learning pipeline.

The next group of files presents the scripts that can be run in Flask. 

process_data.py (link) - This executes the functions presented in the ETL Pipeline Preparation.ipynb file.
train_classifier.py (link) - This executes the functions presented in the ML Pipeline Preparation.ipynb file.
run.py - This file provideds script that runs the web app

Lastly there are two datset files:
disaster_messages.csv (link) - this file contains all of the messages that the model wil be trained on
disaster_categories.csv (link) - here we have data showing which categories each messages relates to

__Usage instructions:__  

To run the ETL and ML pipeline preparation files you will need to download them along with the disaster_messages and disaster_categories files.
For the code to run you will also need to modify the cell that reads in the data in the ETL prepartion file.

To run the .py files you will need to upload them and the data files into a Flask workspace.
To run process_data.py as it is you will need to run the prompt:
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run the train_classifier.py file you will need to execute the prompt:
python models/train_classifier.py data/DisasterResponse.db models/NLPmodel.pkl

Lastly, to use the run.py file you will need to use the prompt:
python app/run.py

Please note that the files will need to be run in this order.

I had some issues with using the IDE provided by Udacity due to the versions of certain packages. As a result to run these files please make sure you are using an up-to-date versions of the joblib and classification_report packages.


__Packages used: There are quite a few packages used in this project.__

Some are standard tools for analysing and visualising numerical data:

pandas
numpy
matplotlib.pyplot
seaborn

Other are tools for analysing text data, mostly derived from nltk:

nltk
nltk.corpus.stopwords
nltk.tokenize.word_tokenize
nltk.stem.WordNetLemmatizer

Some packages were used for specific data preprocessing tasks:

re: this was used to format text data and remove punctuation from numerical values before converting to float
string: provides a convenient way to remove punctuation


__Contact information:__ The maintainer of this project is me - Laurence Durham - contactable at laurence.durham89@gmail.com

__Necessary acknowledgments:__
