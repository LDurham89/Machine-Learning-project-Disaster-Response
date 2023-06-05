import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
#from sklearn.externals import joblib
from sqlalchemy import create_engine



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('NLPproject', engine)

# load model
model = joblib.load("models/NLPmodel.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    X = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    X['related'] = X['related'].map({0: 0, 1: 1, 2: 1})
                
    Num = X.apply(pd.Series.value_counts)
    Q = Num / len(df) *100 
    Q = Q.iloc[1]
    Q = round(Q, 1)
    Q= pd.DataFrame(Q)            
    Q= Q.rename_axis('Category')
    Q.columns = ['Percentage']
                
    g_groups = df.groupby('genre').sum()['request']
    genre_names = list(g_groups.index)
    
    # create visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x=Q.index,
                    y=Q['Percentage']
                )
            ],

            'layout': {
                'title': 'Percentage of message by category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
    {
            'data': [
                Bar(
                    x=genre_names,
                    y=g_groups
                )
            ],

            'layout': {
                'title': 'Number of requests by genre of message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre name"
                }
            }
        }         
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
