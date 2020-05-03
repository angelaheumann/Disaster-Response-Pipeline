import json

import pandas as pd
from sklearn.externals import joblib
from sqlalchemy import create_engine

import plotly
from flask import Flask, jsonify, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar, Heatmap, Pie

app = Flask(__name__)


def tokenize(text):
    '''
    tokenization, lemmatization of the raw text
    INPUT:
        raw text
    OUTPUT:
        cleaned tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster', engine)


# load model
model = joblib.load("../models/classifier.pkl")


@app.route('/')
@app.route('/index')
def index():
    '''
    index webpage displays cool visuals and receives user input text for model
    '''
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    categories = df.drop(['message', 'original', 'genre', 'id'], axis=1)

    # create visuals
    graphs = [

        {
            'data': [
                Pie(
                    values=genre_counts,
                    labels=genre_names,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
            }
        },

        {
            'data': [
                Heatmap(
                    z=categories.values,
                    x=categories.columns,
                    y=categories.index,
                    colorscale='Blues'
                )
            ],

            'layout': {
                'title': 'Distribution of Categories: 1 means belongs to category, 0 means does not belong to category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    '''
    web page that handles user query and displays model results
    '''
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
