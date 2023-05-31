import sys
import pandas as pd
import numpy as np
import sqlite3
import re
from sqlalchemy import create_engine
import os

def load_data(messages_filepath, categories_filepath):
    
 messages = pd.read_csv(messages_filepath)
 categories = pd.read_csv(categories_filepath)
 df = messages.merge(categories, on=('id')) 
 return df, categories


def clean_data(df):
 categories = df['categories'].str.split(";", expand=True)
 row = categories.iloc[[0]]
 category_colnames = row.apply(lambda x: x.str.slice(stop=-2))
 category_colnames = category_colnames.values.tolist()
 categories.columns = category_colnames

 for column in categories:
     categories[column] = categories[column].str.replace("\D+", '')
    
 categories.columns = categories.columns.map('_'.join)
 categories = categories.astype(int)
 df = df.drop('categories', axis = 1)
 df = df.join(categories) 
 df = df.drop_duplicates(subset=['id'])

 return df


def save_data(df, database_filepath): 
    engine = create_engine(database_filepath) 
    df.to_sql('NLPproject', engine, index=False, if_exists='replace')
 

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data("data/disaster_messages.csv", "data/disaster_categories.csv")

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, 'sqlite:///data/DisasterResponse.db')
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()