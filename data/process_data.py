import sys
import pandas as pd
from sqlalchemy import create_engine 

#load messages and categories dataset
def load_data(messages_filepath, categories_filepath):
    #load message dataset
    messages = pd.read_csv(messages_filepath)
    #load categories dataset
    categories=pd.read_csv(categories_filepath)
    # merge messages and categories datasets using the common id
    df_merged = pd.merge(messages,categories,on='id')
    #split the categories into separate category columns, categories dataframe will get updated into including more expanded columns
    categories = categories['categories'].str.split(';', expand = True)
    #select the 1st row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories, which is to extract text part of "related-1" and "request-0"
    category_colnames= row.apply(lambda x: x[:-2])
    #rename the columns of 'categories'
    categories.columns = category_colnames
    #convert category values to just numbers 0 or 1
    for column in categories:
        #set each value to be the last character of the string
        categories[column]= categories[column].astype(str).str[-1:]
        #convert column from string to numeric, then convert non-binary value 2 to 1
        categories[column]=categories[column].astype(int).replace(2,1)
    
    #Replace categories column in df with new category columns.
    # drop the original categories column from `df_merged`
    df_merged.drop(['categories'], axis =1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df_merged = pd.concat([df_merged,categories],axis =1)
    return df_merged


def clean_data(df):
    #remove duplicates
    # check df original shape
    print('Before removing duplicated, df shape: {}'.format(df.shape))
    #check number of duplicates 
    print('duplicated rows before removing: {}'.format(df.duplicated().value_counts()))
    #drop duplicated 
    df.drop_duplicates(inplace=True)
    #check number of duplicates
    print('duplicated rows after removing: {}'.format(df.duplicated().value_counts()))
    return df


def save_data(df, database_filename):
    # stores data in a SQLite database
    engine=create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename,engine,index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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