# Udacity Datascience Nanodegree Project 2: Disaster Response Pipelines

__Project description:__ The second project due as part of Udacity's Data Science Nanodegree involves building a web app that can classify messages to show how they relate to natural disasters.

There are three stages to this project:
1) Extracting and cleaning data
2) Using a machine learning pipeline to create a model for classifying data
3) Deploying the model as a web app using Flask.

The end result is an app that visualises the data and uses the model developed in stage 2 to classify messages.

__File descriptions:__ This project contains several files. The first two files are jupyter notebooks that work through the process of building the ETL and machine learning pipeline, while offering some narrative explaining any difficulties faced and decisions made. These file are:

- ETL Pipeline Preparation.ipynb (https://github.com/LDurham89/Machine-Learning-project-Disaster-Response/blob/main/ETL%20Pipeline%20Preparation.ipynb) - This outlines how the ETL pipeline is made and presents functions that can be used in the web app 

- ML Pipeline Preparation.ipynb (https://github.com/LDurham89/Machine-Learning-project-Disaster-Response/blob/main/ML%20Pipeline%20Preparation.ipynb) - This file does the same for the machine learning pipeline.

The next group of files presents the scripts that can be run in Flask. 

- process_data.py (https://github.com/LDurham89/Machine-Learning-project-Disaster-Response/blob/main/process_data.py) - This executes the functions presented in the ETL Pipeline Preparation.ipynb file.

- train_classifier.py (https://github.com/LDurham89/Machine-Learning-project-Disaster-Response/blob/main/train_classifier.py) - This executes the functions presented in the ML Pipeline Preparation.ipynb file.

- run.py (https://github.com/LDurham89/Machine-Learning-project-Disaster-Response/blob/main/run.py) - This file provideds script that runs the web app

Lastly there are two datset files:
- disaster_messages.csv (https://github.com/LDurham89/Machine-Learning-project-Disaster-Response/blob/main/disaster_messages.csv) - this file contains all of the messages that the model wil be trained on

- disaster_categories.csv (https://github.com/LDurham89/Machine-Learning-project-Disaster-Response/blob/main/disaster_categories.csv) - here we have data showing which categories each messages relates to

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

I had some issues with using the IDE provided by Udacity due to the versions of certain packages. As a result to run these files please make sure you are using up-to-date versions of the joblib and classification_report packages.

I'd also like to note that GridSearchCV can take a significant amount of time to run. As a result in my code I only pass one parameter to the model, but I have included other parameters commented out - just to show how I would have run Grid Search if I had a more powerful machine. 


__Packages used: There are quite a few packages used in this project.__

This project uses packages for a wide variety of tasks.

First are some of the most common general data processing packages, plus re which is useful for editting data:

- pandas
- numpy
- re

The ETL pipeline also used packages to work with SQL:
- sqlite3: allows us to work in SQL
- sqlalchemy - create_engine: creates a bridge to the location where we want to save a database or access an existing database

The ML pipeline uses several NLTK packages to develop the tokenization function and preprocess data:

- nltk.corpus - stopwords
- nltk.tokenize - word_tokenize
- nltk.stem - WordNetLemmatizer

The ML pipeline then uses the following sklearn packages to build the pipleine itself and train the model:

- Pipeline: This provides the basic architecture for building a pipeline
- train_test_split: Splits our data into testing and training subsets
- CountVectorizer
- TfidfTransformer
- MultiOutputClassifier: This is a class that allows us to run a model that performs multiple classification tasks in parellel
- RandomForestClassifier: This is the estimator that we use with the MultiOutputClassifier object
- GridSearchCV: used to find model parameters that optimise performance
- classification_report: The classification report allows us to evaluate the model by providing recall, precision and f1 scores

The last package used in the ML pipeline saves our model:

- pickle: saves the trained model as a pickle file

The last set of packages allows the app to runa nd display the visualisations contained in the app

- json
- plotly
- flask - Flask
- flask - render_template, request, jsonify
- plotly.graph_objs - Bar
- joblib

__Model performance and difficulties faced:__ 

The dataset provided for this project was an interesting one, which provided some real challenges. The main challenge was that the data tended to be unbalanced across most of the categories provided. For some categories this wasn't so bad - e.g. the 'aid related' category applied to about 40% of observations hence allowing enough variation to make decent estimates. On the other hand, the 'child alone' category only had zero values. This meant that the model never saw any messages about children alone and - as you might expect - returned zeros for all measures of the classification report. To some extent this is a tautology - the model made no positive predicitons thus didn't get any correct positive predictions - but the point remains that the model isn't able to identify messges about children alone. 

To some extent this imbalance could be a result of having so many columns. For example, most of the infrastructure related columns were very imbalanced with very few positive cases. While the project rubrik required a model that performs classification for all 36 categories, I think that the model would be more useful if all of the infrastructure related columns were reduced to one 'infrastructure' column. The model would then have a higher absolute number of positive cases to learn from. Admittedly it would be better if were possible to distinguish between each category. If I were doing a similar project in a professional context then I would be keen to investigate ways to secure more data to train the model on, including leveraging any partnerships with other organisations in the sector.

When looking at the classification report it appears that model performance is closely related to the numner of positive values in each category.
The model did well at identifying if messages were relevant to a disaster and (to a lesser extent) if they were aid related - as reflected in the f1 scores close to 1. However, when getting down to individual types of aid required or specific infrastructure issues there tended to be less than 5% positive cases per category. It is also these categories where the model got low f1 scores, for example see thr scores for 'clothing', 'money' and 'missing people'.


__Contact information:__ The maintainer of this project is me - Laurence Durham - contactable at laurence.durham89@gmail.com

__Necessary acknowledgments:__
Udacity GPT was particularly helpful for finding bugs in my code and also navigating the Flask IDE provided by Udacity.

The statology resource below was a useful reference for interpreting the classification reports:
https://www.statology.org/sklearn-classification-report/


