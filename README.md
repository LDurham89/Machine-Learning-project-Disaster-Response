Udacity Datascience Nanodegree Project 2: Disaster Response Pipelines

Project description: 

File descriptions:


The main files of interest for this project are:

Packages used: There are quite a few packages used in this project.

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
nltk.util.ngrams
nltk.probability.FreqDist

Some packages were used for specific data preprocessing tasks:

re: this was used to format text data and remove punctuation from numerical values before converting to float
string: provides a convenient way to remove punctuation
counter: provides a convenient way to count frequency of ngrams
These packages were used for the regression analysis:

statsmodels.api: this contains the linear regression model. I chose this package over sklearns solution as statsmodels allows you to access a report showing the results of the regression
statsmodels.stats.outliers_influence.variance_inflation_factor: this is a useful tool for measuring the degree of multicolinearity in a set of variables
Finally, the package below is useful for export tables. Please note you will need kaleidoscope installed for this to work.

plotly.figure_factory

Summary of the results & issues with analysis: 

Contact information: The maintainer of this project is me - Laurence Durham - contactable at laurence.durham89@gmail.com

Necessary acknowledgments: 
