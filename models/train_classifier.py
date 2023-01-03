import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import sys
import pandas as pd
from sqlalchemy import create_engine, inspect
import nltk
'''
# code snippet to resolve ssl console error
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
'''
nltk.download(['stopwords', 'punkt', 'wordnet'])


# load data from database
def load_data(database_filepath):
    '''
    Loads a pandas Dataframe from a sqlite database
    Args:
    database_filepath: path of the sqlite database
    Returns:
    X: features (data frame)
    Y: target categories (data frame)
    categories: index list with names of categories (series)
    '''
    engine = create_engine(f"sqlite:///{database_filepath}")
    insp = inspect(engine)
    print(insp.get_table_names())

    df = pd.read_sql_table(database_filepath, engine)
    X = df['message']
    Y = df.iloc[:, 4:].fillna(0).astype(int)
    category_names = Y.columns
    return X, Y, category_names


# Tokenization function to process text data of 'message'
'''
In order to shorten the length of the vocabulary, we usually choose the following methods:
 -Ignore case
 -Ignore punctuation
 -Remove meaningless words, such as a the of
 -Fix spelling errors (to implement in next version)
 -Take out the tense (to implement in next version)
 -n-gram vocabulary merge (to implement in next version)
'''


def tokenize(text):
    # regrex to identify url in text
    '''
    A few url patterns identified in message column:
    - http haiti.ushahidi.com
    - http www.youtube.com watch?v=8IySBl2aq A
    - http://xxxx
    - http bit.ly 80fWse #sydney
    - https://
    '''
    url_regex = r'http[s]?[ ]{0,}[:]{0,}[ ]{0,}[//]{0,}(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls in text using regex
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    # remove punctuation
    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    stop_words = set(stopwords.words('English'))
    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        if not tok.lower() in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Creates a pipeline for model training including a GridSearchCv object

    1. CountVectorizer.Count the occurrences of tokens in each document/message
    2. TfidfTransformer. Normalizing and weighting with diminishing importance tokens that occur in the majority of documents/messages
    3. MultioutputClassifier. As there are 36 categories in Y, aka 36 targets, here use MultioutputClassifier as a wrapper to extend classifiers that do not natively support
    multi-target classification. Such classifier includes
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ]
    )
    print(pipeline.get_params().keys())
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # print classification report for positive labels
    Y_pred = model.predict(X_test)
    for i in range(len(Y_test.columns) - 1):
        print("---------------{}----------------".format(category_names[i]))
        print(classification_report(Y_test.values[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Saves model as a .pkl file. Destination is set by model_filepath
    Arguments:
    model: trained sklearn estimator to save
    model_filepath: destination to save model
    '''
    # Save the model into a pickle
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Loads the data, splits it into a train set (80%) and test set(20%), trains the model, use GridSearch to find best parameters,
    evaluates the model on the test set, then save the trained model as a .pkl file
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)
        print(X_train)
        print('Building model...')
        print(Y_train)
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
