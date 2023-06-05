
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

def load_data(filepath):
    engine = create_engine(filepath)
    df = pd.read_sql('SELECT * FROM NLPproject', engine)
    X = df['message'] 
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    #remove punctuation 
    text = re.sub(r'[^\w\s]', '', text) # w matches alphanumeric, s matches whitespace
    
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    
    no_stops = [w for w in tokens if w not in stopwords.words("english")]
        
    tidy_tokens = []
    
    for t in no_stops:
     lemmed = lemmatizer.lemmatize(t).lower().strip()
     tidy_tokens.append(lemmed)
          
    return tidy_tokens


def build_model():   
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfid', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
    #'clf__estimator__bootstrap': [True, False],
    #'clf__estimator__criterion': ['gini', 'entropy', 'log_loss'],
    'tfidf__smooth_idf': [True, False]
}
    cv = GridSearchCV(pipeline, param_grid=parameters) 
    
    return cv   


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns = category_names)
    Y_test = Y_test.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:", accuracy)
    report = classification_report(Y_test, y_pred, output_dict=True, target_names = category_names)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)


def save_model(model, model_filepath):
 pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data('sqlite:///data/DisasterResponse.db')
        Y['related'] = Y['related'].replace([2], 1 )
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, 'models/NLPmodel.pkl')

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
