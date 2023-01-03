"""Module providingFunction printing python version."""
import json
import re
import plotly
import pandas as pd
import numpy as np


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# tokenizer function to tokenize and lemmatize the text, including remove
# urls, punctuation, stopwords


def tokenize(text):
    '''
    tokenizer function to tokenize and lemmatize the text, including remove urls, punctuation, stopwords

    Input:
    text    text in string format
    Output:
    clean_tokens a list of meaningful tokens extracted from text
    '''
    url_regex = r'http[s]?[ ]{0,}[:]{0,}[ ]{0,}[//]{0,}(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls in text using regex
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    # remove punctuation
    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('English'))

    clean_tokens = []
    for tok in tokens:
        if not tok.lower() in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    Displays data visualizations and received user input text for model
    '''
    # extract data needed for visuals
    # 1. data
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # 2. data for distribution count of every category
    df_categories = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    # for category = related, value set is 0,1,2; for category child_alone, there is only 0 ;
    # for other categories, value set is 0,1. Therefore transform values so
    # that sum of columns equals to counts of frequency.
    df_categories = df_categories.replace(
        to_replace='2', value='1').fillna(0).astype(int)
    # Get the non-zero count for each category, sort by descending count number
    category_count = df_categories.sum().sort_values(ascending=False)
    category_names = list(category_count.index)

    # 3. data for distribution of number of categories(labels) of each message
    hist, bin_edges = np.histogram(
        df_categories.sum(
            axis=1), bins=np.arange(
            start=0.5, stop=21.5, step=1))
    bin_grid = np.arange(start=1, stop=21, step=1)

    # 4. data for get top 10 word tokens across all messages and its count
    message_list = list(df['message'])
    vect = CountVectorizer(tokenizer=tokenize)
    message_vectorized = vect.fit_transform(message_list)
    word_list = vect.get_feature_names_out()
    count_list = message_vectorized.toarray().sum(axis=0)

    top10_index = np.argsort(count_list)[::-1][:10]
    top10_word = [word_list[i] for i in top10_index]
    top10_count = [count_list[i] for i in top10_index]

    # create visuals for 4 data sets obtained above
    # it may take around 14 seconds to load the last graph, as tokenize 20k+
    # messages takes a bit time
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
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
                    x=bin_grid,
                    y=hist
                )
            ],

            'layout': {
                'title': 'Distribution of Number of category labels for each message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of labels for each message"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top10_word,
                    y=top10_count
                )
            ],

            'layout': {
                'title': 'Top 10 word tokens and counts',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    ''' function that handles user query and dispalys model results'''
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
    '''entry function to run the app'''
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
