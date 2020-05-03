import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine

warnings.filterwarnings('ignore')

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    '''
    Loads data from the SQLite database
    '''
    # Load dataset from database with read_sql_table
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disaster", engine)
    X = df.message.values
    Y = df.drop(['message', 'original', 'genre', 'id'], axis=1)
    # convert 3 values of related to 2 values
    Y.related = Y.related.apply(lambda x: 1 if x == 2 else x)
    category_names = Y.columns
    return X, Y, category_names


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


def build_model():
    '''
    OUTPUT:
        GridSearchCV
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

        ])),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(penalty='l1', dual=False))))
    ])

    # specify parameters for grid search
    parameters = {
       'features__text_pipeline__vect__max_df': [0.5, 0.7, 1],
       'features__text_pipeline__tfidf__use_idf': [True, False]
    }
    # create grid search object
    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
        model, x test set, y test set, category_names
    Outputs results on the test set
    '''
    y_pred = model.predict(X_test)

    print("Labels:", category_names)
    print("Classification report:\n", classification_report(Y_test.values, y_pred, target_names=category_names))
    print("Accuracy:", (y_pred == Y_test).mean())
    print("Accuracy mean:", round(np.mean((y_pred == Y_test).mean()), 2))
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    '''
    INPUT:
        final model, path where the model should be saved to
    Exports the final model as a pickle file
    '''
    filehandler = open(model_filepath, 'wb')
    pickle.dump(model, filehandler)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
